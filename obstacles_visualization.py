from Pillow import Image


def newImg():
    img = Image.new("RGB", (100, 100))
    img.putpixel((30, 60), (155, 155, 55))
    img.save("sqr.png")

    return img


wallpaper = newImg()
wallpaper.show()
