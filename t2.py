import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px
from util.filter import *
from util.image_manipulation import *

image = plt.imread("./images/frame01284.png")
one_channel = isolate_channel(image, 0)

fft = np.fft.fftshift(np.fft.fft2(one_channel))
magnitude = np.abs(fft)
phase = np.angle(fft)
filtered = band_pass_filter(fft)
ifft_img = np.fft.ifft2(np.fft.ifftshift(filtered))

# fig = make_subplots(
#     rows=2,
#     cols=2,
#     column_widths=[0.5, 0.5],
#     subplot_titles=("Original", "Magnitude", "Filtered", "IFFT"),
#     specs=[
#         [{"type": "Image"}, {"type": "Image"}],
#         [{"type": "Image"}, {"type": "Image"}],
#     ],
# )
# fig.add_trace(px.imshow(image).data[0], row=1, col=1)
# mul = 255 / np.max(np.log(magnitude))
# fig.add_trace(px.imshow(np.log(magnitude) * mul).data[0], row=1, col=2)
# fig.add_trace(px.imshow(np.log(np.abs(filtered)) * mul).data[0], row=2, col=1)
# fig.add_trace(
#     px.imshow(np.abs(ifft_img) * (255 / np.max(np.abs(ifft_img)))).data[0], row=2, col=2
# )
# fig.update_layout(
#     xaxis=dict(constrain="domain"),
#     yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
#     xaxis2=dict(constrain="domain"),
#     yaxis2=dict(scaleanchor="x2", scaleratio=1, autorange="reversed"),
#     xaxis3=dict(constrain="domain"),
#     yaxis3=dict(scaleanchor="x3", scaleratio=1, autorange="reversed"),
#     xaxis4=dict(constrain="domain"),
#     yaxis4=dict(scaleanchor="x4", scaleratio=1, autorange="reversed"),
#     coloraxis={"colorscale": "gray"},
# )
# fig.show()


fig, ax = plt.subplots(1, 3)

ax[0].imshow(image)
ax[1].imshow(np.log(magnitude))
ax[2].imshow(np.log(np.abs(filtered)))

# ax[0][0].imshow(image)
# ax[0][0].set_title("Original")

# ax[0][1].imshow(np.log(magnitude), cmap="gray")
# ax[0][1].set_title("FFT")

# ax[1][0].imshow(np.log(np.abs(filtered)), cmap="gray")
# ax[1][0].set_title("Edge detected")

# ax[1][1].imshow(np.abs(ifft_img), cmap="gray")
# ax[1][1].set_title("Inverse FFT")

plt.imsave("./images/edge_detected.png", np.abs(ifft_img), cmap="gray")

fig.canvas.manager.full_screen_toggle()
plt.show()
