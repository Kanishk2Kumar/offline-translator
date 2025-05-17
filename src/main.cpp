#include <../portaudio/include/portaudio.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include "../whisper.cpp/include/whisper.h"

#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 512
#define NUM_SECONDS 5  // Buffer duration for each transcription chunk

std::vector<float> audioBuffer;
bool recording = true;

static int recordCallback(const void* input, void*, unsigned long frameCount,
                          const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void*) {
    const float* in = (const float*)input;
    if (in == nullptr) return paContinue;

    for (unsigned long i = 0; i < frameCount; ++i) {
        audioBuffer.push_back(in[i]);
    }

    return paContinue;
}

void transcribe(const std::vector<float>& audio, struct whisper_context* ctx) {
    if (audio.empty()) return;

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;

    if (whisper_full(ctx, params, audio.data(), audio.size()) != 0) {
        std::cerr << "Whisper failed!\n";
        return;
    }

    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        std::cout << text << std::endl;
    }
}

int main() {
    std::cout << "[+] Initializing PortAudio...\n";
    Pa_Initialize();

    PaStream* stream;
    Pa_OpenDefaultStream(&stream, 1, 0, paFloat32, SAMPLE_RATE,
                         FRAMES_PER_BUFFER, recordCallback, nullptr);
    Pa_StartStream(stream);

    std::cout << "[+] Loading whisper model...\n";
    struct whisper_context* ctx = whisper_init_from_file("models/ggml-base.en.bin");

    std::cout << "[+] Recording...\n";

    while (recording) {
        std::this_thread::sleep_for(std::chrono::seconds(NUM_SECONDS));
        std::vector<float> chunk(audioBuffer.begin(), audioBuffer.end());
        audioBuffer.clear();

        std::thread(transcribe, chunk, ctx).detach();
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    whisper_free(ctx);

    return 0;
}
