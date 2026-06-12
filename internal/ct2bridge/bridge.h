#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ct2_model ct2_model;
typedef struct ct2_encoder_output ct2_encoder_output;

typedef struct {
    int32_t* sequences_ids;
    size_t sequences_count;
    float score;
    float no_speech_prob;
    char* error;
} ct2_generate_result;

typedef struct {
    char* language;
    float probability;
    char* error;
} ct2_detect_result;

ct2_model* ct2_model_load(const char* path, const char* device, const char* compute_type);
void ct2_model_free(ct2_model* m);
bool ct2_model_is_multilingual(ct2_model* m);
int32_t ct2_model_n_mels(ct2_model* m);
const char* ct2_last_error(void);

ct2_encoder_output* ct2_encode(
    ct2_model* m,
    const float* mel, size_t n_mels, size_t n_frames);
void ct2_encoder_output_free(ct2_encoder_output* e);

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
    int max_initial_timestamp_index);

ct2_detect_result ct2_detect_language(
    ct2_model* m,
    ct2_encoder_output* encoder_output);

void ct2_generate_result_free(ct2_generate_result* r);
void ct2_detect_result_free(ct2_detect_result* r);

#ifdef __cplusplus
}
#endif
