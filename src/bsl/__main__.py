import argparse
import json
import sys
# Keep the health check usable when native vision/ML imports are broken or slow.
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='doctor':
    from .doctor import main as doctor_main
    sys.exit(doctor_main(sys.argv[2:]))
from .predict import Predictor, DEFAULT_MODEL
from .media import frames

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='doctor':
        from .doctor import main as doctor
        return doctor(argv[1:])
    parser = argparse.ArgumentParser(description='Bangla sign research app: images, video, camera and setup diagnostics')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor',help='Check dependency versions and model integrity')
    for command in ('predict', 'webcam', 'gui'):
        p = sub.add_parser(command)
        p.add_argument('--model', default=str(DEFAULT_MODEL))
        p.add_argument('--labels')
        if command == 'predict':
            p.add_argument('image', nargs='?')
        if command == 'gui':
            p.add_argument('--mode',choices=['letters','personal','sentences'],default='letters')
        if command != 'gui':
            source = p.add_mutually_exclusive_group()
            source.add_argument('--video')
            source.add_argument('--images')
            p.add_argument('--output', help='Write JSONL predictions')
            p.add_argument('--camera', type=int, default=0)
            display=p.add_mutually_exclusive_group()
            display.add_argument('--preview',action='store_true',help='Show annotated video window')
            display.add_argument('--no-preview',action='store_true',help='Headless capture')
            p.add_argument('--full-frame',action='store_true',help='Classify whole camera frame instead of centre guide')
            p.add_argument('--compose',action='store_true',help='Accumulate stable isolated letters; space/backspace/c clear text')
            p.add_argument('--min-confidence',type=float,default=.6)
    args = parser.parse_args(argv)
    try:
        if args.command == 'gui':
            from .gui import launch
            return launch(args.model, args.labels,args.mode)
        if args.command == 'predict' and not (args.image or args.video or args.images):
            parser.error('predict requires an image, --images or --video')
        if args.command == 'predict' and args.image and (args.video or args.images):
            parser.error('Choose exactly one input')
        predictor = Predictor(args.model, args.labels)
        output = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
        try:
            if args.command=='predict' and args.image:
                print(json.dumps(predictor(args.image),ensure_ascii=False),file=output,flush=True)
            else:
                import cv2
                from .preview import hand_crop,annotate
                from .text import TextComposer,is_blank
                camera=args.video is None and args.images is None
                show=not args.no_preview and (args.preview or camera)
                composer=TextComposer(min_confidence=args.min_confidence) if args.compose else None
                try:
                    for source,frame in frames(video=args.video,images=args.images,camera=args.camera):
                        sample,box=hand_crop(frame) if camera and not args.full_frame else (frame,None)
                        result=predictor(sample)
                        row={'source':source,**result}
                        if composer:
                            row['text']=composer.update(result,blank=is_blank(sample))
                        print(json.dumps(row,ensure_ascii=False),file=output,flush=True)
                        if show:
                            cv2.imshow('Bangla Sign Lab | q quit, SPACE word, BACKSPACE undo, c clear',
                                       annotate(frame,result,composer.text if composer else 'Fit your hand inside the guide',box))
                            key=cv2.waitKey(1)&255
                            if key in (27,ord('q')):
                                break
                            if composer:
                                if key==32: composer.space()
                                elif key in (8,127): composer.backspace()
                                elif key==ord('c'): composer.clear()
                finally:
                    if show:
                        cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass
        finally:
            if args.output:
                output.close()
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')

if __name__ == '__main__':
    sys.exit(main())
