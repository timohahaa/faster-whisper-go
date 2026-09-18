#include "bridge.h"

#include <cstring>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <ctranslate2/devices.h>
#include <ctranslate2/models/whisper.h>
#include <ctranslate2/replica_pool.h>
#include <ctranslate2/storage_view.h>

struct ct2_model {
    std::unique_ptr<ctranslate2::models::Whisper> whisper;
    // when the model spans more than one
    // CUDA device, the encoder output is moved to host memory so any replica
    // (on any device) can consume it, since we don't know which device will run
    // the next step. Keeps single-device runs on the GPU (no copy).
    bool encoder_to_cpu = false;
};

struct ct2_encoder_output {
    ctranslate2::StorageView view;
};

namespace {

char* copy_string(const std::string& value) {
    char* out = static_cast<char*>(std::malloc(value.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

ctranslate2::Device parse_device(const char* device) {
    if (device == nullptr || device[0] == '\0') {
        return ctranslate2::Device::CPU;
    }
    return ctranslate2::str_to_device(device);
}

ctranslate2::ComputeType parse_compute_type(const char* compute_type) {
    if (compute_type == nullptr || compute_type[0] == '\0') {
        return ctranslate2::ComputeType::DEFAULT;
    }
    return ctranslate2::str_to_compute_type(compute_type);
}

ctranslate2::StorageView make_mel_features(
    const float* mel, size_t n_mels, size_t n_frames) {
    std::vector<float> data(n_mels * n_frames);
    std::memcpy(data.data(), mel, data.size() * sizeof(float));
    return ctranslate2::StorageView(
        {1, static_cast<ctranslate2::dim_t>(n_mels), static_cast<ctranslate2::dim_t>(n_frames)},
        data,
        ctranslate2::Device::CPU);
}

ctranslate2::StorageView make_mel_features_batch(
    const float* mel, size_t batch_size, size_t n_mels, size_t n_frames) {
    size_t total = batch_size * n_mels * n_frames;
    std::vector<float> data(total);
    std::memcpy(data.data(), mel, total * sizeof(float));
    return ctranslate2::StorageView(
        {static_cast<ctranslate2::dim_t>(batch_size),
         static_cast<ctranslate2::dim_t>(n_mels),
         static_cast<ctranslate2::dim_t>(n_frames)},
        data,
        ctranslate2::Device::CPU);
}

ctranslate2::models::WhisperOptions make_whisper_options(
    int beam_size, int best_of,
    float patience, float length_penalty,
    float repetition_penalty, int no_repeat_ngram_size,
    int max_length, bool suppress_blank,
    float sampling_temperature,
    const int32_t* suppress_tokens, size_t suppress_tokens_count) {
    ctranslate2::models::WhisperOptions options;
    options.patience = patience > 0 ? patience : 1.0f;
    options.length_penalty = length_penalty > 0 ? length_penalty : 1.0f;
    options.repetition_penalty = repetition_penalty > 0 ? repetition_penalty : 1.0f;
    options.no_repeat_ngram_size =
        no_repeat_ngram_size > 0 ? static_cast<size_t>(no_repeat_ngram_size) : 0;
    options.max_length = max_length > 0 ? static_cast<size_t>(max_length) : 448;
    options.suppress_blank = suppress_blank;
    options.return_scores = true;
    options.return_no_speech_prob = true;

    if (sampling_temperature > 0) {
        options.beam_size = 1;
        options.num_hypotheses = best_of > 0 ? static_cast<size_t>(best_of) : 1;
        options.sampling_topk = 0;
        options.sampling_temperature = sampling_temperature;
    } else {
        options.beam_size = beam_size > 0 ? static_cast<size_t>(beam_size) : 1;
        options.num_hypotheses = 1;
    }

    if (suppress_tokens != nullptr && suppress_tokens_count > 0) {
        std::vector<int> stoks(suppress_tokens_count);
        for (size_t i = 0; i < suppress_tokens_count; ++i) {
            stoks[i] = static_cast<int>(suppress_tokens[i]);
        }
        options.suppress_tokens = std::move(stoks);
    }

    return options;
}

ct2_batch_generate_result make_batch_generate_error(const char* message) {
    ct2_batch_generate_result result{};
    result.error = copy_string(message);
    return result;
}

ct2_batch_align_result make_batch_align_error(const char* message) {
    ct2_batch_align_result result{};
    result.error = copy_string(message);
    return result;
}

ct2_batch_detect_result make_batch_detect_error(const char* message) {
    ct2_batch_detect_result result{};
    result.error = copy_string(message);
    return result;
}

void set_error_out(char** error_out, const char* message) {
    if (error_out != nullptr) {
        *error_out = copy_string(message);
    }
}

ct2_generate_result make_generate_error(const char* message) {
    ct2_generate_result result{};
    result.error = copy_string(message);
    return result;
}

ct2_detect_result make_detect_error(const char* message) {
    ct2_detect_result result{};
    result.error = copy_string(message);
    return result;
}

}  // namespace

extern "C" {

ct2_model* ct2_model_load(
    const char* path, const char* device, const char* compute_type,
    const int* device_index, size_t device_index_count,
    int intra_threads, int inter_threads,
    char** error_out) {
    if (path == nullptr || path[0] == '\0') {
        set_error_out(error_out, "model path is required");
        return nullptr;
    }

    try {
        std::vector<int> dev_indices;
        if (device_index != nullptr && device_index_count > 0) {
            dev_indices.assign(device_index, device_index + device_index_count);
        } else {
            dev_indices = {0};
        }

        // Decide whether the encoder output must be moved to the host. Base this
        // on distinct devices from the original list (before the inter_threads
        // expansion below, which duplicates indices): several replicas on the
        // same card share device memory, so keeping the output on the GPU is
        // safe there; only crossing distinct devices requires the host round-trip.
        const ctranslate2::Device dev = parse_device(device);
        const std::set<int> distinct_devices(dev_indices.begin(), dev_indices.end());
        const bool encoder_to_cpu =
            dev == ctranslate2::Device::CUDA && distinct_devices.size() > 1;

        // inter_threads > 1 means multiple replicas per device.
        // CTranslate2 maps one replica per device_index entry,
        // so we repeat each index to get the desired replica count.
        if (inter_threads > 1) {
            std::vector<int> expanded;
            expanded.reserve(dev_indices.size() * inter_threads);
            for (int idx : dev_indices) {
                for (int r = 0; r < inter_threads; ++r) {
                    expanded.push_back(idx);
                }
            }
            dev_indices = std::move(expanded);
        }

        ctranslate2::ReplicaPoolConfig pool_config;
        if (intra_threads > 0) pool_config.num_threads_per_replica = intra_threads;

        auto model = std::make_unique<ct2_model>();
        model->encoder_to_cpu = encoder_to_cpu;
        model->whisper = std::make_unique<ctranslate2::models::Whisper>(
            path,
            dev,
            parse_compute_type(compute_type),
            dev_indices,
            /*tensor_parallel=*/false,
            pool_config);
        return model.release();
    } catch (const std::exception& e) {
        set_error_out(error_out, e.what());
        return nullptr;
    }
}

void ct2_model_free(ct2_model* m) {
    delete m;
}

bool ct2_model_is_multilingual(ct2_model* m) {
    if (m == nullptr || m->whisper == nullptr) {
        return false;
    }
    return m->whisper->is_multilingual();
}

int32_t ct2_model_n_mels(ct2_model* m) {
    if (m == nullptr || m->whisper == nullptr) {
        return 0;
    }
    return static_cast<int32_t>(m->whisper->n_mels());
}

ct2_encoder_output* ct2_encode(
    ct2_model* m,
    const float* mel, size_t n_mels, size_t n_frames,
    char** error_out) {
    if (m == nullptr || m->whisper == nullptr) {
        set_error_out(error_out, "model is null");
        return nullptr;
    }
    if (mel == nullptr || n_mels == 0 || n_frames == 0) {
        set_error_out(error_out, "mel spectrogram is required");
        return nullptr;
    }

    try {
        ctranslate2::StorageView features = make_mel_features(mel, n_mels, n_frames);
        auto future = m->whisper->encode(features, /*to_cpu=*/m->encoder_to_cpu);
        ctranslate2::StorageView encoded = future.get();
        auto out = new ct2_encoder_output{std::move(encoded)};
        return out;
    } catch (const std::exception& e) {
        set_error_out(error_out, e.what());
        return nullptr;
    }
}

void ct2_encoder_output_free(ct2_encoder_output* e) {
    if (e == nullptr) {
        return;
    }
    // The encoder output may hold GPU memory. Freeing it runs the StorageView
    // destructor on the caller's (Go) thread, whose current CUDA device is not
    // necessarily the one the buffer lives on. cudaSetDevice is per-thread, so
    // set the buffer's device around the free; otherwise a buffer on device >0
    // gets freed under device 0 and corrupts the context (invalid argument /
    // illegal memory access). On CPU this is a no-op path.
    if (e->view.device() == ctranslate2::Device::CUDA) {
        const ctranslate2::ScopedDeviceSetter scope(
            e->view.device(), e->view.device_index());
        delete e;
        return;
    }
    delete e;
}

ct2_generate_result ct2_generate(
    ct2_model* m,
    ct2_encoder_output* encoder_output,
    const int32_t* prompt_tokens, size_t prompt_count,
    int beam_size, int best_of,
    float patience, float length_penalty,
    float repetition_penalty, int no_repeat_ngram_size,
    int max_length,
    bool suppress_blank,
    float sampling_temperature,
    const int32_t* suppress_tokens, size_t suppress_tokens_count,
    int max_initial_timestamp_index) {
    if (m == nullptr || m->whisper == nullptr) {
        return make_generate_error("model is null");
    }
    if (encoder_output == nullptr) {
        return make_generate_error("encoder output is required");
    }
    if (prompt_tokens == nullptr && prompt_count > 0) {
        return make_generate_error("prompt tokens are required");
    }

    try {
        std::vector<size_t> prompt(prompt_count);
        for (size_t i = 0; i < prompt_count; ++i) {
            prompt[i] = static_cast<size_t>(prompt_tokens[i]);
        }

        ctranslate2::models::WhisperOptions options = make_whisper_options(
            beam_size, best_of, patience, length_penalty,
            repetition_penalty, no_repeat_ngram_size,
            max_length, suppress_blank, sampling_temperature,
            suppress_tokens, suppress_tokens_count);
        options.max_initial_timestamp_index =
            max_initial_timestamp_index >= 0 ? static_cast<size_t>(max_initial_timestamp_index) : 50;

        const std::vector<std::vector<size_t>> prompts = {prompt};
        std::vector<std::future<ctranslate2::models::WhisperGenerationResult>> futures =
            m->whisper->generate(encoder_output->view, prompts, options);

        if (futures.empty()) {
            return make_generate_error("generate returned no results");
        }

        const std::vector<ctranslate2::models::WhisperGenerationResult> results = {
            futures.front().get()};

        const auto& result = results.front();
        if (result.sequences_ids.empty()) {
            return make_generate_error("generate returned no token sequences");
        }

        const std::vector<size_t>& sequence_raw = result.sequences_ids.front();
        ct2_generate_result out{};
        out.sequences_count = sequence_raw.size();
        out.sequences_ids = static_cast<int32_t*>(std::malloc(
            out.sequences_count * sizeof(int32_t)));
        if (out.sequences_ids == nullptr) {
            return make_generate_error("failed to allocate token sequence");
        }
        for (size_t i = 0; i < out.sequences_count; ++i) {
            out.sequences_ids[i] = static_cast<int32_t>(sequence_raw[i]);
        }

        if (result.has_scores()) {
            out.score = result.scores.front();
        }
        out.no_speech_prob = result.no_speech_prob;
        return out;
    } catch (const std::exception& e) {
        return make_generate_error(e.what());
    }
}

ct2_detect_result ct2_detect_language(
    ct2_model* m,
    ct2_encoder_output* encoder_output) {
    if (m == nullptr || m->whisper == nullptr) {
        return make_detect_error("model is null");
    }
    if (encoder_output == nullptr) {
        return make_detect_error("encoder output is required");
    }

    try {
        std::vector<std::future<std::vector<std::pair<std::string, float>>>> futures =
            m->whisper->detect_language(encoder_output->view);

        if (futures.empty()) {
            return make_detect_error("detect_language returned no results");
        }

        const std::vector<std::pair<std::string, float>> results = futures.front().get();

        if (results.empty()) {
            return make_detect_error("detect_language returned no results");
        }

        const auto& best = results.front();
        ct2_detect_result out{};
        out.language = copy_string(best.first);
        out.probability = best.second;
        if (out.language == nullptr) {
            return make_detect_error("failed to allocate language result");
        }
        return out;
    } catch (const std::exception& e) {
        return make_detect_error(e.what());
    }
}

ct2_align_result ct2_align(
    ct2_model* m,
    ct2_encoder_output* encoder_output,
    const int32_t* start_sequence, size_t start_sequence_count,
    const int32_t* text_tokens, size_t text_tokens_count,
    size_t num_frames,
    int median_filter_width) {
    ct2_align_result out{};

    if (m == nullptr || m->whisper == nullptr) {
        out.error = copy_string("model is null");
        return out;
    }
    if (encoder_output == nullptr) {
        out.error = copy_string("encoder output is required");
        return out;
    }
    if (text_tokens == nullptr || text_tokens_count == 0) {
        out.error = copy_string("text tokens are required");
        return out;
    }

    try {
        std::vector<size_t> start_seq(start_sequence_count);
        for (size_t i = 0; i < start_sequence_count; ++i) {
            start_seq[i] = static_cast<size_t>(start_sequence[i]);
        }

        std::vector<size_t> tokens(text_tokens_count);
        for (size_t i = 0; i < text_tokens_count; ++i) {
            tokens[i] = static_cast<size_t>(text_tokens[i]);
        }

        std::vector<std::vector<size_t>> token_batches = {tokens};
        std::vector<size_t> num_frames_vec = {num_frames};

        int filter_width = median_filter_width > 0 ? median_filter_width : 7;

        auto futures = m->whisper->align(
            encoder_output->view,
            start_seq,
            token_batches,
            num_frames_vec,
            static_cast<ctranslate2::dim_t>(filter_width));

        if (futures.empty()) {
            out.error = copy_string("align returned no results");
            return out;
        }

        std::vector<ctranslate2::models::WhisperAlignmentResult> results;
        results.reserve(futures.size());
        for (auto& f : futures) {
            results.push_back(f.get());
        }

        if (results.empty() || results.front().alignments.empty()) {
            out.error = copy_string("align returned no results");
            return out;
        }

        const auto& alignment = results.front();

        // Return text_token_probs.
        size_t n_tokens = alignment.text_token_probs.size();
        out.num_tokens = n_tokens;
        out.text_token_probs = static_cast<float*>(
            std::malloc(n_tokens * sizeof(float)));
        if (out.text_token_probs == nullptr) {
            out.error = copy_string("failed to allocate text_token_probs");
            return out;
        }
        for (size_t i = 0; i < n_tokens; ++i) {
            out.text_token_probs[i] = alignment.text_token_probs[i];
        }

        // Return raw alignment pairs (text_index, time_index).
        size_t n_align = alignment.alignments.size();
        out.num_alignments = n_align;
        out.text_indices = static_cast<int32_t*>(
            std::malloc(n_align * sizeof(int32_t)));
        out.time_indices = static_cast<int32_t*>(
            std::malloc(n_align * sizeof(int32_t)));
        if (out.text_indices == nullptr || out.time_indices == nullptr) {
            out.error = copy_string("failed to allocate alignment indices");
            return out;
        }
        for (size_t i = 0; i < n_align; ++i) {
            out.text_indices[i] = static_cast<int32_t>(alignment.alignments[i].first);
            out.time_indices[i] = static_cast<int32_t>(alignment.alignments[i].second);
        }

        return out;
    } catch (const std::exception& e) {
        out.error = copy_string(e.what());
        return out;
    }
}

void ct2_generate_result_free(ct2_generate_result* r) {
    if (r == nullptr) {
        return;
    }
    std::free(r->sequences_ids);
    std::free(r->error);
    r->sequences_ids = nullptr;
    r->error = nullptr;
    r->sequences_count = 0;
}

void ct2_detect_result_free(ct2_detect_result* r) {
    if (r == nullptr) {
        return;
    }
    std::free(r->language);
    std::free(r->error);
    r->language = nullptr;
    r->error = nullptr;
    r->probability = 0;
}

void ct2_align_result_free(ct2_align_result* r) {
    if (r == nullptr) {
        return;
    }
    std::free(r->text_token_probs);
    std::free(r->text_indices);
    std::free(r->time_indices);
    std::free(r->error);
    r->text_token_probs = nullptr;
    r->text_indices = nullptr;
    r->time_indices = nullptr;
    r->error = nullptr;
    r->num_tokens = 0;
    r->num_alignments = 0;
}

/* ---- Batched API ---- */

ct2_encoder_output* ct2_encode_batch(
    ct2_model* m,
    const float* mel, size_t batch_size, size_t n_mels, size_t n_frames,
    char** error_out) {
    if (m == nullptr || m->whisper == nullptr) {
        set_error_out(error_out, "model is null");
        return nullptr;
    }
    if (mel == nullptr || batch_size == 0 || n_mels == 0 || n_frames == 0) {
        set_error_out(error_out, "mel spectrogram batch is required");
        return nullptr;
    }

    try {
        ctranslate2::StorageView features =
            make_mel_features_batch(mel, batch_size, n_mels, n_frames);
        auto future = m->whisper->encode(features, /*to_cpu=*/m->encoder_to_cpu);
        ctranslate2::StorageView encoded = future.get();
        auto out = new ct2_encoder_output{std::move(encoded)};
        return out;
    } catch (const std::exception& e) {
        set_error_out(error_out, e.what());
        return nullptr;
    }
}

ct2_batch_generate_result ct2_generate_batch(
    ct2_model* m,
    ct2_encoder_output* encoder_output,
    const int32_t** prompt_tokens, const size_t* prompt_counts,
    size_t batch_size,
    int beam_size, int best_of,
    float patience, float length_penalty,
    float repetition_penalty, int no_repeat_ngram_size,
    int max_length,
    bool suppress_blank,
    float sampling_temperature,
    const int32_t* suppress_tokens, size_t suppress_tokens_count) {
    if (m == nullptr || m->whisper == nullptr) {
        return make_batch_generate_error("model is null");
    }
    if (encoder_output == nullptr) {
        return make_batch_generate_error("encoder output is required");
    }
    if (batch_size == 0) {
        return make_batch_generate_error("batch_size must be > 0");
    }

    try {
        std::vector<std::vector<size_t>> prompts(batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            size_t count = prompt_counts[b];
            prompts[b].resize(count);
            for (size_t i = 0; i < count; ++i) {
                prompts[b][i] = static_cast<size_t>(prompt_tokens[b][i]);
            }
        }

        ctranslate2::models::WhisperOptions options = make_whisper_options(
            beam_size, best_of, patience, length_penalty,
            repetition_penalty, no_repeat_ngram_size,
            max_length, suppress_blank, sampling_temperature,
            suppress_tokens, suppress_tokens_count);

        auto futures = m->whisper->generate(
            encoder_output->view, prompts, options);

        if (futures.size() != batch_size) {
            return make_batch_generate_error("generate returned unexpected number of results");
        }

        ct2_batch_generate_result out{};
        out.batch_size = batch_size;
        out.sequences_ids = static_cast<int32_t**>(
            std::calloc(batch_size, sizeof(int32_t*)));
        out.sequences_counts = static_cast<size_t*>(
            std::calloc(batch_size, sizeof(size_t)));
        out.scores = static_cast<float*>(
            std::calloc(batch_size, sizeof(float)));
        out.no_speech_probs = static_cast<float*>(
            std::calloc(batch_size, sizeof(float)));

        if (!out.sequences_ids || !out.sequences_counts ||
            !out.scores || !out.no_speech_probs) {
            ct2_batch_generate_result_free(&out);
            return make_batch_generate_error("failed to allocate batch result");
        }

        for (size_t b = 0; b < batch_size; ++b) {
            auto result = futures[b].get();

            if (result.sequences_ids.empty() || result.sequences_ids.front().empty()) {
                out.sequences_counts[b] = 0;
                continue;
            }

            const auto& seq = result.sequences_ids.front();
            out.sequences_counts[b] = seq.size();
            out.sequences_ids[b] = static_cast<int32_t*>(
                std::malloc(seq.size() * sizeof(int32_t)));
            if (out.sequences_ids[b] == nullptr) {
                ct2_batch_generate_result_free(&out);
                return make_batch_generate_error("failed to allocate token sequence");
            }
            for (size_t i = 0; i < seq.size(); ++i) {
                out.sequences_ids[b][i] = static_cast<int32_t>(seq[i]);
            }

            if (result.has_scores()) {
                out.scores[b] = result.scores.front();
            }
            out.no_speech_probs[b] = result.no_speech_prob;
        }

        return out;
    } catch (const std::exception& e) {
        return make_batch_generate_error(e.what());
    }
}

void ct2_batch_generate_result_free(ct2_batch_generate_result* r) {
    if (r == nullptr) {
        return;
    }
    if (r->sequences_ids != nullptr) {
        for (size_t i = 0; i < r->batch_size; ++i) {
            std::free(r->sequences_ids[i]);
        }
        std::free(r->sequences_ids);
    }
    std::free(r->sequences_counts);
    std::free(r->scores);
    std::free(r->no_speech_probs);
    std::free(r->error);
    *r = ct2_batch_generate_result{};
}

ct2_batch_align_result ct2_align_batch(
    ct2_model* m,
    ct2_encoder_output* encoder_output,
    const int32_t* start_sequence, size_t start_sequence_count,
    const int32_t** text_tokens, const size_t* text_tokens_counts,
    const size_t* num_frames, size_t batch_size,
    int median_filter_width) {
    if (m == nullptr || m->whisper == nullptr) {
        return make_batch_align_error("model is null");
    }
    if (encoder_output == nullptr) {
        return make_batch_align_error("encoder output is required");
    }
    if (batch_size == 0) {
        return make_batch_align_error("batch_size must be > 0");
    }

    try {
        std::vector<size_t> start_seq(start_sequence_count);
        for (size_t i = 0; i < start_sequence_count; ++i) {
            start_seq[i] = static_cast<size_t>(start_sequence[i]);
        }

        // Per-item token sequences. Empty sequences are allowed and produce an
        // empty alignment for that item (chunks without decoded segments).
        std::vector<std::vector<size_t>> token_batches(batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            size_t count = text_tokens_counts[b];
            token_batches[b].resize(count);
            for (size_t i = 0; i < count; ++i) {
                token_batches[b][i] = static_cast<size_t>(text_tokens[b][i]);
            }
        }

        std::vector<size_t> num_frames_vec(batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            num_frames_vec[b] = num_frames[b];
        }

        int filter_width = median_filter_width > 0 ? median_filter_width : 7;

        auto futures = m->whisper->align(
            encoder_output->view,
            start_seq,
            token_batches,
            num_frames_vec,
            static_cast<ctranslate2::dim_t>(filter_width));

        if (futures.size() != batch_size) {
            return make_batch_align_error(
                "align returned unexpected number of results");
        }

        ct2_batch_align_result out{};
        out.batch_size = batch_size;
        out.num_tokens = static_cast<size_t*>(
            std::calloc(batch_size, sizeof(size_t)));
        out.text_token_probs = static_cast<float**>(
            std::calloc(batch_size, sizeof(float*)));
        out.num_alignments = static_cast<size_t*>(
            std::calloc(batch_size, sizeof(size_t)));
        out.text_indices = static_cast<int32_t**>(
            std::calloc(batch_size, sizeof(int32_t*)));
        out.time_indices = static_cast<int32_t**>(
            std::calloc(batch_size, sizeof(int32_t*)));

        if (!out.num_tokens || !out.text_token_probs || !out.num_alignments ||
            !out.text_indices || !out.time_indices) {
            ct2_batch_align_result_free(&out);
            return make_batch_align_error("failed to allocate batch align result");
        }

        for (size_t b = 0; b < batch_size; ++b) {
            auto result = futures[b].get();
            if (result.alignments.empty()) {
                continue;  // leave zeros for this item
            }

            size_t n_tokens = result.text_token_probs.size();
            out.num_tokens[b] = n_tokens;
            if (n_tokens > 0) {
                out.text_token_probs[b] = static_cast<float*>(
                    std::malloc(n_tokens * sizeof(float)));
                if (out.text_token_probs[b] == nullptr) {
                    ct2_batch_align_result_free(&out);
                    return make_batch_align_error(
                        "failed to allocate text_token_probs");
                }
                for (size_t i = 0; i < n_tokens; ++i) {
                    out.text_token_probs[b][i] = result.text_token_probs[i];
                }
            }

            size_t n_align = result.alignments.size();
            out.num_alignments[b] = n_align;
            out.text_indices[b] = static_cast<int32_t*>(
                std::malloc(n_align * sizeof(int32_t)));
            out.time_indices[b] = static_cast<int32_t*>(
                std::malloc(n_align * sizeof(int32_t)));
            if (out.text_indices[b] == nullptr || out.time_indices[b] == nullptr) {
                ct2_batch_align_result_free(&out);
                return make_batch_align_error(
                    "failed to allocate alignment indices");
            }
            for (size_t i = 0; i < n_align; ++i) {
                out.text_indices[b][i] =
                    static_cast<int32_t>(result.alignments[i].first);
                out.time_indices[b][i] =
                    static_cast<int32_t>(result.alignments[i].second);
            }
        }

        return out;
    } catch (const std::exception& e) {
        return make_batch_align_error(e.what());
    }
}

void ct2_batch_align_result_free(ct2_batch_align_result* r) {
    if (r == nullptr) {
        return;
    }
    if (r->text_token_probs != nullptr) {
        for (size_t i = 0; i < r->batch_size; ++i) {
            std::free(r->text_token_probs[i]);
        }
        std::free(r->text_token_probs);
    }
    if (r->text_indices != nullptr) {
        for (size_t i = 0; i < r->batch_size; ++i) {
            std::free(r->text_indices[i]);
        }
        std::free(r->text_indices);
    }
    if (r->time_indices != nullptr) {
        for (size_t i = 0; i < r->batch_size; ++i) {
            std::free(r->time_indices[i]);
        }
        std::free(r->time_indices);
    }
    std::free(r->num_tokens);
    std::free(r->num_alignments);
    std::free(r->error);
    *r = ct2_batch_align_result{};
}

ct2_batch_detect_result ct2_detect_language_batch(
    ct2_model* m,
    ct2_encoder_output* encoder_output) {
    if (m == nullptr || m->whisper == nullptr) {
        return make_batch_detect_error("model is null");
    }
    if (encoder_output == nullptr) {
        return make_batch_detect_error("encoder output is required");
    }

    try {
        std::vector<std::future<std::vector<std::pair<std::string, float>>>> futures =
            m->whisper->detect_language(encoder_output->view);

        size_t batch_size = futures.size();
        if (batch_size == 0) {
            return make_batch_detect_error("detect_language returned no results");
        }

        ct2_batch_detect_result out{};
        out.batch_size = batch_size;
        out.languages = static_cast<char**>(std::calloc(batch_size, sizeof(char*)));
        out.probabilities = static_cast<float*>(std::calloc(batch_size, sizeof(float)));
        if (!out.languages || !out.probabilities) {
            ct2_batch_detect_result_free(&out);
            return make_batch_detect_error("failed to allocate batch detect result");
        }

        for (size_t b = 0; b < batch_size; ++b) {
            auto results = futures[b].get();
            if (results.empty()) {
                continue;  // leave nullptr/zero for this item
            }
            const auto& best = results.front();
            out.languages[b] = copy_string(best.first);
            out.probabilities[b] = best.second;
            if (out.languages[b] == nullptr) {
                ct2_batch_detect_result_free(&out);
                return make_batch_detect_error("failed to allocate language result");
            }
        }

        return out;
    } catch (const std::exception& e) {
        return make_batch_detect_error(e.what());
    }
}

void ct2_batch_detect_result_free(ct2_batch_detect_result* r) {
    if (r == nullptr) {
        return;
    }
    if (r->languages != nullptr) {
        for (size_t i = 0; i < r->batch_size; ++i) {
            std::free(r->languages[i]);
        }
        std::free(r->languages);
    }
    std::free(r->probabilities);
    std::free(r->error);
    *r = ct2_batch_detect_result{};
}

}  // extern "C"
