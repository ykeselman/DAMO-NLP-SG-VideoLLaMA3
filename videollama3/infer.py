import sys
sys.path.append('./')
from videollama3 import disable_torch_init, model_init, mm_infer


def main():
    disable_torch_init()

    modal = 'text'
    modal_path = None
    instruct = 'What is the color of bananas?'

    # valid output: The woman is wearing a stylish black leather coat over a red dress, paired with black boots. She carries a small black handbag and has sunglasses on her head. 2691
    modal = 'image'
    modal_path = 'assets/sora.png'
    instruct = 'What is the woman wearing?'

    # valid output: The cat is lying on its back, appearing relaxed and content.
    modal = 'video'
    modal_path = 'assets/cat_and_chicken.mp4'
    instruct = 'What is the cat doing?'

    model_path = ''

    model, processor, tokenizer = model_init(model_path)
    output = mm_infer(
        processor[modal](modal_path, fps=1, max_frames=768) if modal != 'text' else None,
        instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
    print(output)


if __name__ == '__main__':
    main()
