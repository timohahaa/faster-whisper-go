#include "bridge.h"

#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>

struct ct2_model {
    std::unique_ptr<ctranslate2::models::Whisper> whisper;
};

struct ct2_encoder_output {
    ctranslate2::StorageView view;
};

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::exception& e) {
    g_last_error = e.what();
}

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

const char* ct2_last_error(void) {
    return g_last_error.c_str();
}

ct2_model* ct2_model_load(
    const char* path, const char* device, const char* compute_type) {
    g_last_error.clear();
    if (path == nullptr || path[0] == '\0') {
        g_last_error = "model path is required";
        return nullptr;
    }

    try {
        auto model = std::make_unique<ct2_model>();
        model->whisper = std::make_unique<ctranslate2::models::Whisper>(
            path,
            parse_device(device),
            parse_compute_type(compute_type));
        return model.release();
    } catch (const std::exception& e) {
        set_last_error(e);
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
    const float* mel, size_t n_mels, size_t n_frames) {
    g_last_error.clear();
    if (m == nullptr || m->whisper == nullptr) {
        g_last_error = "model is null";
        return nullptr;
    }
    if (mel == nullptr || n_mels == 0 || n_frames == 0) {
        g_last_error = "mel spectrogram is required";
        return nullptr;
    }

    try {
        ctranslate2::StorageView features = make_mel_features(mel, n_mels, n_frames);
        auto future = m->whisper->encode(features, /*to_cpu=*/false);
        ctranslate2::StorageView encoded = future.get();
        auto out = new ct2_encoder_output{std::move(encoded)};
        return out;
    } catch (const std::exception& e) {
        set_last_error(e);
        return nullptr;
    }
}

void ct2_encoder_output_free(ct2_encoder_output* e) {
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
    bool suppress_blank, bool return_scores,
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

        ctranslate2::models::WhisperOptions options;
        options.patience = patience > 0 ? patience : 1.0f;
        options.length_penalty = length_penalty > 0 ? length_penalty : 1.0f;
        options.repetition_penalty = repetition_penalty > 0 ? repetition_penalty : 1.0f;
        options.no_repeat_ngram_size =
            no_repeat_ngram_size > 0 ? static_cast<size_t>(no_repeat_ngram_size) : 0;
        options.max_length = max_length > 0 ? static_cast<size_t>(max_length) : 448;
        options.suppress_blank = suppress_blank;
        options.return_scores = return_scores;
        options.return_no_speech_prob = true;
        options.max_initial_timestamp_index =
            max_initial_timestamp_index >= 0 ? static_cast<size_t>(max_initial_timestamp_index) : 50;

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
        std::vector<int> start_seq(start_sequence_count);
        for (size_t i = 0; i < start_sequence_count; ++i) {
            start_seq[i] = static_cast<int>(start_sequence[i]);
        }

        std::vector<int> tokens(text_tokens_count);
        for (size_t i = 0; i < text_tokens_count; ++i) {
            tokens[i] = static_cast<int>(text_tokens[i]);
        }

        std::vector<std::vector<int>> token_batches = {tokens};
        std::vector<size_t> num_frames_vec = {num_frames};

        int filter_width = median_filter_width > 0 ? median_filter_width : 7;

        std::vector<ctranslate2::models::WhisperAlignmentResult> results =
            m->whisper->align(
                encoder_output->view,
                start_seq,
                token_batches,
                num_frames_vec,
                filter_width);

        if (results.empty() || results.front().alignments.empty()) {
            out.error = copy_string("align returned no results");
            return out;
        }

        const auto& alignment = results.front();
        size_t n_tokens = alignment.alignments.size();
        size_t n_frames_out = 0;
        if (n_tokens > 0) {
            n_frames_out = alignment.alignments.front().size();
        }

        out.num_tokens = n_tokens;
        out.num_frames = n_frames_out;
        out.weights = static_cast<float*>(
            std::malloc(n_tokens * n_frames_out * sizeof(float)));
        if (out.weights == nullptr) {
            out.error = copy_string("failed to allocate alignment weights");
            return out;
        }

        for (size_t t = 0; t < n_tokens; ++t) {
            const auto& row = alignment.alignments[t];
            for (size_t f = 0; f < n_frames_out; ++f) {
                out.weights[t * n_frames_out + f] = row[f];
            }
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
    std::free(r->weights);
    std::free(r->error);
    r->weights = nullptr;
    r->error = nullptr;
    r->num_tokens = 0;
    r->num_frames = 0;
}

}  // extern "C"
