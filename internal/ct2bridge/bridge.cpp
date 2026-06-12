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

ct2_generate_result ct2_generate(
    ct2_model* m,
    const float* mel, size_t n_mels, size_t n_frames,
    const int32_t* prompt_tokens, size_t prompt_count,
    int beam_size, int best_of, float patience, float length_penalty,
    float repetition_penalty, int no_repeat_ngram_size,
    int max_length, bool suppress_blank, bool return_scores) {
    if (m == nullptr || m->whisper == nullptr) {
        return make_generate_error("model is null");
    }
    if (mel == nullptr || n_mels == 0 || n_frames == 0) {
        return make_generate_error("mel spectrogram is required");
    }
    if (prompt_tokens == nullptr && prompt_count > 0) {
        return make_generate_error("prompt tokens are required");
    }

    try {
        ctranslate2::StorageView features = make_mel_features(mel, n_mels, n_frames);

        std::vector<size_t> prompt(prompt_count);
        for (size_t i = 0; i < prompt_count; ++i) {
            prompt[i] = static_cast<size_t>(prompt_tokens[i]);
        }

        ctranslate2::models::WhisperOptions options;
        options.beam_size = beam_size > 0 ? static_cast<size_t>(beam_size) : 1;
        options.num_hypotheses = best_of > 0 ? static_cast<size_t>(best_of) : 1;
        options.patience = patience > 0 ? patience : 1.0f;
        options.length_penalty = length_penalty > 0 ? length_penalty : 1.0f;
        options.repetition_penalty = repetition_penalty > 0 ? repetition_penalty : 1.0f;
        options.no_repeat_ngram_size =
            no_repeat_ngram_size > 0 ? static_cast<size_t>(no_repeat_ngram_size) : 0;
        options.max_length = max_length > 0 ? static_cast<size_t>(max_length) : 448;
        options.suppress_blank = suppress_blank;
        options.return_scores = return_scores;
        options.return_no_speech_prob = true;

        const std::vector<std::vector<size_t>> prompts = {prompt};
        std::vector<std::future<ctranslate2::models::WhisperGenerationResult>> futures =
            m->whisper->generate(features, prompts, options);

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
    const float* mel, size_t n_mels, size_t n_frames) {
    if (m == nullptr || m->whisper == nullptr) {
        return make_detect_error("model is null");
    }
    if (mel == nullptr || n_mels == 0 || n_frames == 0) {
        return make_detect_error("mel spectrogram is required");
    }

    try {
        ctranslate2::StorageView features = make_mel_features(mel, n_mels, n_frames);
        std::vector<std::future<std::vector<std::pair<std::string, float>>>> futures =
            m->whisper->detect_language(features);

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

}  // extern "C"
