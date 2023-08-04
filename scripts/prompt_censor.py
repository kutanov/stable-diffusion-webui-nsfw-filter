from transformers import pipeline

pipe = pipeline("text-classification", model="Zlatislav/NSFW-Prompt-Detector")

# path_to_model = os.path.join(os.path.dirname(__file__), '../models/prompt_detector.bin')
# checkpoint = torch.load(path_to_model)
# models.load_state_dict = model.load_state_dict(state_dict)

def is_prompt_safe(prompt):
    result = pipe(prompt)
    print(result)
    # if result[0]['label'] == "NSFW" and result[0]['score'] > 0.62:
    #         return False
    # if result[0]['label'] == "SFW" and result[0]['score'] < 0.7:
    #         return False
    return True