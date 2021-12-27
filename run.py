import os, glob, shutil, argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image,ImageDraw,ImageFont
from skimage.transform import resize

def make_dir(path, refresh=False):

    try:
        os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def min_max_norm(x):

    return (x - x.min() + 1e-31) / (x.max() - x.min() + 1e-31)

def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def min_max_norm(x):

    return (x - x.min() + 1e-31) / (x.max() - x.min() + 1e-31)

def gen_font(tmp_char, font, chara_size, bound=50):

    font = ImageFont.truetype(font=font, size=chara_size)
    x, y = font.getsize(tmp_char)
    theImage = Image.new('RGB', (x+bound*2, y+bound*2), color='white')
    theDrawPad = ImageDraw.Draw(theImage)
    theDrawPad.text((bound, bound), tmp_char, font=font, fill='black')

    tmp_img = np.asarray(theImage)
    tmp_img = rgb2gray(rgb=tmp_img)
    tmp_img = min_max_norm(x=tmp_img)

    idx_hs, idx_he, idx_ws, idx_we = 0, 0, 0, 0
    val_avg = np.average(tmp_img)
    for idx_h in range(tmp_img.shape[0]):
        if(idx_hs == 0 and np.average(tmp_img[idx_h, :]) < 1):
            idx_hs = idx_h
    for idx_h in list(range(tmp_img.shape[0]))[::-1]:
        if(idx_he == 0 and np.average(tmp_img[idx_h, :]) < 1):
            idx_he = idx_h
            break
    for idx_w in range(tmp_img.shape[1]):
        if(idx_ws == 0 and np.average(tmp_img[:, idx_w]) < 1):
            idx_ws = idx_w
    for idx_w in list(range(tmp_img.shape[1]))[::-1]:
        if(idx_we == 0 and np.average(tmp_img[:, idx_w]) < 1):
            idx_we = idx_w
            break

    height = idx_he - idx_hs
    width = idx_we - idx_ws

    return tmp_img, height, width, idx_hs, idx_he, idx_ws, idx_we

def main():

    list_ttf = sorted_list(path=os.path.join(FLAGS.dir_ttf, '*.ttf'))

    img_size, chara_size = 224, 224

    ftxt = open(FLAGS.path_txt, 'r')
    contents = ftxt.readlines()
    ftxt.close()

    characters = ""
    for idx_c, content in enumerate(contents):
        characters += content.replace('\n', '')
    print(characters)

    save_dir = FLAGS.dir_save
    make_dir(path=save_dir, refresh=True)
    save_dir_npy = "%s_npy" %(save_dir)
    make_dir(path=save_dir_npy, refresh=True)

    for idx_ttf, path_ttf in enumerate(list_ttf):

        save_subdir = path_ttf.split('/')[-1].split('.')[0]
        make_dir(path=os.path.join(save_dir, save_subdir), refresh=False)
        make_dir(path=os.path.join(save_dir_npy, save_subdir), refresh=False)

        tmp_chara_size = chara_size
        for idx_check in range(2):
            confirm_font, confirm_cnt = False, 0
            while(True):
                for idx_char, tmp_char in enumerate(characters):

                    tmp_img, height, width, idx_hs, idx_he, idx_ws, idx_we = \
                        gen_font(tmp_char=tmp_char, font=path_ttf, chara_size=tmp_chara_size, bound=50)

                    if(height <= img_size and width <= img_size):
                        confirm_cnt += 1
                        if(confirm_cnt >= len(characters)):
                            confirm_font = True
                    else:
                        tmp_chara_size -= 1
                        confirm_cnt = 0

                    if(idx_check == 1):
                        aln_img = np.ones((256, 256))
                        aln_hs = int((aln_img.shape[0]/2) - (height/2))
                        aln_he = aln_hs + height
                        aln_ws = int((aln_img.shape[1]/2) - (width/2))
                        aln_we = aln_ws + width
                        aln_img[aln_hs:aln_he, aln_ws:aln_we] = tmp_img[idx_hs:idx_he, idx_ws:idx_we]

                        save_name = "%s_%s.png" %(tmp_char, ord(tmp_char))

                        if(img_size != FLAGS.size_img):
                            aln_img = resize(aln_img, (FLAGS.size_img, FLAGS.size_img))

                        plt.imsave( \
                            os.path.join(save_dir, save_subdir, save_name), aln_img, cmap='gray')
                        np.save( \
                            os.path.join(save_dir_npy, save_subdir, save_name.replace('png', 'npy')), aln_img)
                if(confirm_font):
                    break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_ttf', type=str, default='ttfs', help='')
    parser.add_argument('--dir_save', type=str, default='dataset_glyph', help='')
    parser.add_argument('--path_txt', type=str, default='text.txt', help='')
    parser.add_argument('--size_img', type=int, default=224, help='')

    FLAGS, unparsed = parser.parse_known_args()

    main()
