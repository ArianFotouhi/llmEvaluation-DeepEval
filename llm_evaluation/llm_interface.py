from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class MyLlama:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device=None):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.device = 0 if torch.cuda.is_available() else -1 if device is None else device

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )

    def generate_rude_response(self, prompt):
        rude_prompt = (
            "You are a sarcastic assistant. Be unhelpful, and overly honest.\n\n"
            f"User: {prompt}\nAssistant:"
        )
        response = self.generator(rude_prompt, return_full_text=False)[0]["generated_text"]
        return response.strip()
