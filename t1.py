import numpy as np
import matplotlib.pyplot as plt


def extract_channel(img, channel: int):
    """Extract the specified channel of the input image. and set other channels to 0."""
    if channel not in range(3):
        raise ValueError("channel must be 0, 1, or 2.")
    channel_img = np.copy(img)
    for i in range(3):
        if i != channel:
            channel_img[:, :, i] = 0
    return channel_img


def isolate_channel(img, channel: int):
    """Isolate the specified channel of the input image."""
    if channel not in range(3):
        raise ValueError("channel must be 0, 1, or 2.")
    channel_img = np.copy(img)
    return channel_img[:, :, channel]


def fft2(img):
    """Compute the 2D FFT of the input image."""
    ft = np.fft.fft2(img)
    return np.fft.fftshift(ft)


def ifft2(img):
    """Compute the 2D inverse FFT of the input image."""
    ift = np.fft.ifft2(img)
    ift = np.fft.fftshift(ift)
    return ift.real


image = plt.imread("./images/bright.png")
channels = [
    extract_channel(image, 0),
    extract_channel(image, 1),
    extract_channel(image, 2),
]
cal_channel = [
    isolate_channel(image, 0),
    isolate_channel(image, 1),
    isolate_channel(image, 2),
]

fig, ax = plt.subplots(1, 4)
ax[0].imshow(image)
ax[0].set_title("Original")
ax[0].axis("off")

fft_channel = [0, 0, 0]

for i in range(3):
    fft_channel[i] = fft2(cal_channel[i])
    ax[i + 1].imshow(np.log(np.abs(fft_channel[i])), cmap="gray")
    ax[i + 1].set_title(f"FFT of channel {i}")
    ax[i + 1].axis("off")
    # plt.imsave(
    #     f"./images/fft_channel_{i}.png",
    #     fft_channel[i],
    # )

test = plt.imread("./images/fft_channel_0.png")
print(test.shape)
print(fft_channel[0])
print(test[0, 0])

plt.show()
ifft_channel = [plt.imread(f"./images/fft_channel_{i}.png") for i in range(3)]
print(ifft_channel[0].shape)

for i in range(3):
    plt.subplot(222 + i)
    ifft = ifft2(ifft_channel[i][:, :, 0])
    plt.imshow(ifft)
    plt.axis("off")

plt.show()
