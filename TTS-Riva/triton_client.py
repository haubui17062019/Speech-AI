from nemo.collections.tts.models import FastPitchModel
# import tritonclient.grpc.aio
import tritonclient.grpc as grpcclient
# from tritonclient.utils import np_to_triton_dtype
import logging
import soundfile as sf

def main():
    MODEL_NAME = "tts"
    URL = "0.0.0.0:8187"
    client = grpcclient.InferenceServerClient(URL)
    spec_generator = FastPitchModel.restore_from("tts_en_fastpitch_align_ipa.nemo", map_location="cpu")
    spec_generator.eval().cuda()
    tokens = spec_generator.parse("FastPitch is a fully-parallel text-to-speech model based on FastSpeech,").to('cpu').numpy()

    inputs = [grpcclient.InferInput("inputs", tokens.shape, "INT64")]
    inputs[0].set_data_from_numpy(tokens)

    outputs = [grpcclient.InferRequestedOutput("audio")]
    # outputs.append(grpcclient.InferRequestedOutput('output'))

    results = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
        client_timeout=None)
    output = results.as_numpy('audio')[0]
    print(output)
    sf.write("output_triton.wav", output, 22050)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

