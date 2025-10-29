import os
import json
import torch
import numpy as np
from PIL import Image
from klarity import UncertaintyEstimator
from klarity.core.analyzer import EnhancedVLMAnalyzer
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, LogitsProcessorList

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize VLM model
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    output_attentions=True,
    low_cpu_mem_usage=True
)

processor = AutoProcessor.from_pretrained(model_id)

# Create estimator with EnhancedVLMAnalyzer
estimator = UncertaintyEstimator(
    top_k=100,
    analyzer=EnhancedVLMAnalyzer(
        min_token_prob=0.01,
        insight_model="your together.ai model",
        insight_api_key="your together.ai api key",
        vision_config=model.config.vision_config,
        use_cls_token=True
    ),
)

uncertainty_processor = estimator.get_logits_processor()

# Set up generation for the example
image_path = "klarity_Experiment_1/Normal_dots.png"
question = "What do you see?"
image = Image.open(image_path)

# Prepare input with image and text
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"}
        ]
    }
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process inputs
inputs = processor(
    images=image,
    text=prompt,
    return_tensors='pt'
)

try:
    # Generate with uncertainty and attention analysis
    generation_output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        logits_processor=LogitsProcessorList([uncertainty_processor]),
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
        use_cache=True
    )

    # Analyze the generation - now includes both images and enhanced analysis
    result = estimator.analyze_generation(
        generation_output=generation_output,
        model=model,
        tokenizer=processor,
        processor=uncertainty_processor,
        prompt=question,
        image=image  # Image is required for enhanced analysis
    )

    # Get generated text
    input_length = inputs.input_ids.shape[1]
    generated_sequence = generation_output.sequences[0][input_length:]
    generated_text = processor.decode(generated_sequence, skip_special_tokens=True)

    print(f"\nQuestion: {question}")
    print(f"Generated answer: {generated_text}")

    # Token Analysis
    print("\nDetailed Token Analysis:")
    for idx, metrics in enumerate(result.token_metrics):
        print(f"\nStep {idx}:")
        print(f"Raw entropy: {metrics.raw_entropy:.4f}")
        print(f"Semantic entropy: {metrics.semantic_entropy:.4f}")
        print("Top 3 predictions:")
        for i, pred in enumerate(metrics.token_predictions[:3], 1):
            print(f"  {i}. {pred.token} (prob: {pred.probability:.4f})")

    # Define your output directory
    output_dir = "/your/path/for/results/goes/here"
    os.makedirs(output_dir, exist_ok=True)

    # Save comprehensive insight
    overall_insight_path = os.path.join(output_dir, "overall_insight.json")
    
    with open(overall_insight_path, "w") as f:
        json.dump(result.overall_insight, f, indent=2)
        
    print(f"Comprehensive Analysis saved to {overall_insight_path}")

    # Get attention arrays
    cumulative_attention = result.attention_data.cumulative_attention
    token_attentions = result.attention_data.token_attentions

    # Save cumulative attention
    if cumulative_attention is not None:
        cumulative_attention_path = os.path.join(output_dir, "cumulative_attention.npy")
        np.save(cumulative_attention_path, cumulative_attention)
        print(f"Cumulative attention saved to {cumulative_attention_path}")

    # Save individual token attentions
    if token_attentions:
        for idx, ta in enumerate(token_attentions):
            token = ta["token"]
            attention_grid = np.array(ta["attention_grid"])

            # sanitize token for filename in case it has weird chars like "/" etc.
            safe_token = str(token).replace("/", "_").replace("\\", "_").replace(" ", "_")

            token_attention_path = os.path.join(
                output_dir,
                f"attention_{idx}_{safe_token}.npy"
            )

            np.save(token_attention_path, attention_grid)
            print(f"Token attention for '{token}' saved to {token_attention_path}")

except Exception as e:
    print(f"Error during generation: {str(e)}")
    import traceback
    traceback.print_exc()
