from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image

def setup_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    # model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
    return [processor,model]

def caption(image,processor,model,text=''):
    raw_image = image.copy()
    if text == '':
        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt",max_new_tokens=50).to("cuda")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)
    else:
        # conditional image captioning
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        return processor.decode(out[0], skip_special_tokens=True)


if __name__ == '__main__':
    processor,model = setup_model()
    raw_image = Image.open('./Ratnadeep.jpg')
    print(caption(raw_image,processor,model))
