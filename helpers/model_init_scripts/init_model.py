import argparse
import os

from transformers import AutoConfig, AutoTokenizer, logging
from parler_tts.dac_wrapper.modeling_dac import DACModel, DACConfig
from parler_tts import ParlerTTSDecoderConfig, ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_directory", type=str, help="Directory where to save the model and the decoder.")
    parser.add_argument("--text_model", default=None, type=str, help="Repository id or path to the text encoder.")
    parser.add_argument("--audio_model", type=str, help="Repository id or path to the audio encoder.")
    parser.add_argument("--tokenizer_model", default=None, type=str, help="Repository id or path to the tokenizer.")
    parser.add_argument("--additional_token_num", default=64, type=int, help="Number of additional tokens to add to the tokenizer.")
    parser.add_argument('--token', type=str, required=True,
                      help='HuggingFace API token')

    args = parser.parse_args()

    text_model = args.text_model
    encodec_version = args.audio_model
    
    
    if text_model is not None:
        t5 = AutoConfig.from_pretrained(text_model)
        vocab_size = t5.vocab_size
        args.tokenizer_model = text_model
        
    encodec, kwargs_audio_encoder = DACConfig.from_pretrained(args.audio_model, return_unused_kwargs=True)
    audio_encoder = DACModel.from_pretrained(
        args.audio_model, config=encodec, **kwargs_audio_encoder
    )

    encodec_vocab_size = encodec.codebook_size
    num_codebooks = encodec.num_codebooks
    print("num_codebooks", num_codebooks)
    
    if args.tokenizer_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, token=args.token)
        tokenizer_vocab_size = tokenizer.vocab_size
        
        # Add new tokens
        new_tokens = ["[parler_tts_eos]", "[parler_tts_bos]", "[male]", "[female]"]
        
        # Add [10s] to [100s] tokens in increments of 10
        new_tokens.extend([f"[{i}s]" for i in range(10, 101, 10)])
        
        additional_tokens = [f"[speaker_{i}]" for i in range(args.additional_token_num - len(new_tokens))]
        
        new_tokens.extend(additional_tokens)
        tokenizer.add_tokens(new_tokens)
        
        vocab_size = tokenizer.vocab_size
        original_vocab_size = tokenizer_vocab_size
        pad_token_id = tokenizer.convert_tokens_to_ids("[parler_tts_eos]") if tokenizer is not None else original_vocab_size
        eos_token_id = tokenizer.convert_tokens_to_ids("[parler_tts_eos]") if tokenizer is not None else original_vocab_size
        bos_token_id = tokenizer.convert_tokens_to_ids("[parler_tts_bos]") if tokenizer is not None else original_vocab_size
        
    vocab_size = vocab_size + args.additional_token_num
        
    decoder_config = ParlerTTSDecoderConfig(
        vocab_size=encodec_vocab_size + 64,  # + 64 instead of +1 to have a multiple of 64
        max_position_embeddings=4096,  # 30 s = 2580
        num_hidden_layers=24,
        ffn_dim=4096,
        num_attention_heads=16,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pad_token_id=encodec_vocab_size,
        eos_token_id=encodec_vocab_size,
        bos_token_id=encodec_vocab_size + 1,
        num_codebooks=num_codebooks,
    )

    decoder = ParlerTTSForCausalLM(decoder_config)
    decoder.save_pretrained(os.path.join(args.save_directory, "decoder"))

    model = ParlerTTSForConditionalGeneration.from_sub_models_pretrained(
        text_encoder_pretrained_model_name_or_path=None,
        audio_encoder_pretrained_model_name_or_path=encodec_version,
        decoder_pretrained_model_name_or_path=os.path.join(args.save_directory, "decoder"),
        vocab_size=vocab_size,
        use_text_encoder=False,
        use_audio_encoder=True,
        tokenizer_class=args.tokenizer_model,
    )

    # set the appropriate bos/pad token ids
    model.generation_config.decoder_start_token_id = encodec_vocab_size + 1
    model.generation_config.pad_token_id = encodec_vocab_size
    model.generation_config.eos_token_id = encodec_vocab_size

    # set other default generation config params
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True  # True

    model.config.pad_token_id = encodec_vocab_size
    model.config.decoder_start_token_id = encodec_vocab_size + 1
    
    # Print the number of parameters in the model in millions
    num_parameters = model.num_parameters()
    print(f"Number of parameters in the model: {num_parameters / 1_000_000:.2f}M")

    model.save_pretrained(os.path.join(args.save_directory, "parler-tts-untrained-600M/"))
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(args.save_directory, "parler-tts-untrained-600M/"))
