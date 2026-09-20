import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def carregar_rgb(caminho):
    img_bgr = cv2.imread(caminho)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível abrir: {caminho}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def exibir(titulo, imagem, cmap=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(imagem, cmap=cmap)
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def resize_nearest(img, nova_altura, nova_largura):
    altura, largura = img.shape[:2]
    if img.ndim == 2:
        saida = np.zeros((nova_altura, nova_largura), dtype=img.dtype)
    else:
        saida = np.zeros((nova_altura, nova_largura, img.shape[2]), dtype=img.dtype)

    for y in range(nova_altura):
        origem_y = min(int(y * altura / nova_altura), altura - 1)
        for x in range(nova_largura):
            origem_x = min(int(x * largura / nova_largura), largura - 1)
            saida[y, x] = img[origem_y, origem_x]
    return saida


def resize_bilinear(img, nova_altura, nova_largura):
    altura, largura = img.shape[:2]
    canais = 1 if img.ndim == 2 else img.shape[2]

    if canais == 1:
        saida = np.zeros((nova_altura, nova_largura), dtype=np.float32)
    else:
        saida = np.zeros((nova_altura, nova_largura, canais), dtype=np.float32)

    escala_y = (altura - 1) / max(nova_altura - 1, 1)
    escala_x = (largura - 1) / max(nova_largura - 1, 1)

    for y in range(nova_altura):
        src_y = y * escala_y
        y0 = int(np.floor(src_y))
        y1 = min(y0 + 1, altura - 1)
        dy = src_y - y0

        for x in range(nova_largura):
            src_x = x * escala_x
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, largura - 1)
            dx = src_x - x0

            p00 = img[y0, x0].astype(np.float32)
            p01 = img[y0, x1].astype(np.float32)
            p10 = img[y1, x0].astype(np.float32)
            p11 = img[y1, x1].astype(np.float32)

            topo = (1 - dx) * p00 + dx * p01
            baixo = (1 - dx) * p10 + dx * p11
            saida[y, x] = (1 - dy) * topo + dy * baixo

    return np.clip(saida, 0, 255).astype(np.uint8)


def transladar(img, dx, dy):
    altura, largura = img.shape[:2]
    saida = np.zeros_like(img)

    for y in range(altura):
        for x in range(largura):
            novo_x = x + dx
            novo_y = y + dy
            if 0 <= novo_x < largura and 0 <= novo_y < altura:
                saida[novo_y, novo_x] = img[y, x]
    return saida


def rotacionar_nearest(img, angulo_graus):
    altura, largura = img.shape[:2]
    saida = np.zeros_like(img)

    cx = (largura - 1) / 2
    cy = (altura - 1) / 2
    angulo = np.deg2rad(angulo_graus)
    cos_a = np.cos(angulo)
    sin_a = np.sin(angulo)

    for y2 in range(altura):
        for x2 in range(largura):
            x = x2 - cx
            y = y2 - cy
            origem_x = cos_a * x + sin_a * y + cx
            origem_y = -sin_a * x + cos_a * y + cy
            ox = int(round(origem_x))
            oy = int(round(origem_y))

            if 0 <= ox < largura and 0 <= oy < altura:
                saida[y2, x2] = img[oy, ox]
    return saida


def quantizar(img, bits):
    niveis = 2 ** bits
    indices = np.round((img.astype(np.float32) / 255.0) * (niveis - 1))
    quantizada = np.round(indices * (255.0 / (niveis - 1)))
    return np.clip(quantizada, 0, 255).astype(np.uint8)


def histograma_manual(cinza):
    hist = np.zeros(256, dtype=np.int64)
    for valor in cinza.ravel():
        hist[int(valor)] += 1
    return hist


def equalizar_histograma_manual(cinza):
    hist = histograma_manual(cinza)
    total = cinza.size
    cdf = np.zeros(256, dtype=np.float64)
    acumulado = 0

    for i in range(256):
        acumulado += hist[i]
        cdf[i] = acumulado / total

    mapa = np.round(cdf * 255).astype(np.uint8)
    saida = np.zeros_like(cinza)
    altura, largura = cinza.shape

    for y in range(altura):
        for x in range(largura):
            saida[y, x] = mapa[cinza[y, x]]

    return saida, histograma_manual(saida)


def questao_1():
    img = carregar_rgb("Lenna.png")
    altura, largura, canais = img.shape

    print("QUESTÃO 1")
    print(f"Dimensões: {largura} x {altura}")
    print(f"Canais: {canais}")

    menor = resize_nearest(img, max(1, altura // 3), max(1, largura // 3))
    exibir("Lenna - original", img)
    exibir("Lenna - matriz 3 vezes menor", menor)

    dpi_original = 96.0
    try:
        with Image.open("Lenna.png") as im:
            dpi_meta = im.info.get("dpi")
            if dpi_meta and dpi_meta[0] > 0:
                dpi_original = float(dpi_meta[0])
    except Exception:
        pass

    dpi_novo = 25.0
    nova_largura = max(1, round(largura * dpi_novo / dpi_original))
    nova_altura = max(1, round(altura * dpi_novo / dpi_original))
    img_25dpi = resize_nearest(img, nova_altura, nova_largura)

    print(f"DPI original utilizado: {dpi_original:.2f}")
    print(f"Dimensões em 25 DPI: {nova_largura} x {nova_altura}")
    exibir("Lenna - reamostrada para 25 DPI", img_25dpi)


def questao_2():
    img = carregar_rgb("Lenna.png")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for bits, ax in zip(range(1, 9), axes.ravel()):
        q = quantizar(img, bits)
        ax.imshow(q)
        ax.set_title(f"{bits} bit(s) - {2 ** bits} níveis")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def questao_3():
    degrade = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    exibir("Degradê de 8 bits - 256 níveis de cinza", degrade, cmap="gray")


def questao_4():
    tamanho = 300
    vermelho = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
    verde = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)
    azul = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)

    vermelho[:, :, 0] = 255
    verde[:, :, 1] = 255
    azul[:, :, 2] = 255

    ciano = np.clip(verde.astype(np.int16) + azul.astype(np.int16), 0, 255).astype(np.uint8)
    magenta = np.clip(vermelho.astype(np.int16) + azul.astype(np.int16), 0, 255).astype(np.uint8)
    amarelo = np.clip(vermelho.astype(np.int16) + verde.astype(np.int16), 0, 255).astype(np.uint8)
    branco = np.clip(vermelho.astype(np.int16) + verde.astype(np.int16) + azul.astype(np.int16), 0, 255).astype(np.uint8)

    imagens = [
        ("Vermelho", vermelho), ("Verde", verde), ("Azul", azul),
        ("Ciano", ciano), ("Magenta", magenta), ("Amarelo", amarelo), ("Branco", branco)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, (nome, imagem) in enumerate(imagens):
        axes.ravel()[i].imshow(imagem)
        axes.ravel()[i].set_title(nome)
        axes.ravel()[i].axis("off")
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    plt.show()


def questao_6(caminho="Lenna.png"):
    img = carregar_rgb(caminho)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    zero = np.zeros_like(r)

    imagens = [
        ("Canal R", np.stack([r, zero, zero], axis=2)),
        ("Canal G", np.stack([zero, g, zero], axis=2)),
        ("Canal B", np.stack([zero, zero, b], axis=2)),
        ("R + B", np.stack([r, zero, b], axis=2)),
        ("R + G", np.stack([r, g, zero], axis=2)),
        ("B + G", np.stack([zero, g, b], axis=2)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (titulo, imagem) in zip(axes.ravel(), imagens):
        ax.imshow(imagem)
        ax.set_title(titulo)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (titulo, canal) in zip(axes, [("H - Matiz", h), ("S - Saturação", s), ("V - Valor/Brilho", v)]):
        ax.imshow(canal, cmap="gray")
        ax.set_title(titulo)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def questao_7():
    img = carregar_rgb("L-RES.png")
    altura, largura = img.shape[:2]

    transladada = transladar(img, dx=6, dy=5)
    redimensionada = resize_nearest(img, altura * 4, largura * 4)
    rotacionada = rotacionar_nearest(redimensionada, 30)

    exibir("Original", img)
    exibir("Translação", transladada)
    exibir("Redimensionamento sem interpolação", redimensionada)
    exibir("Imagem redimensionada e rotacionada", rotacionada)

    nova_altura = rotacionada.shape[0] * 2
    nova_largura = rotacionada.shape[1] * 2

    nearest = resize_nearest(rotacionada, nova_altura, nova_largura)
    bilinear = resize_bilinear(rotacionada, nova_altura, nova_largura)
    bicubica = cv2.resize(rotacionada, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (titulo, imagem) in zip(axes, [("Nearest Neighbor", nearest), ("Bilinear", bilinear), ("Bicúbica", bicubica)]):
        ax.imshow(imagem)
        ax.set_title(titulo)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def questao_8():
    img = carregar_rgb("CONTRAST.png")
    cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = histograma_manual(cinza)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(256), hist, width=1.0)
    plt.title("Histograma original")
    plt.xlabel("Nível de cinza")
    plt.ylabel("Quantidade de pixels")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.show()

    equalizada, hist_eq = equalizar_histograma_manual(cinza)
    exibir("Imagem original em escala de cinza", cinza, cmap="gray")
    exibir("Imagem equalizada", equalizada, cmap="gray")

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(256), hist_eq, width=1.0)
    plt.title("Histograma equalizado")
    plt.xlabel("Nível de cinza")
    plt.ylabel("Quantidade de pixels")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    questao_1()
    questao_2()
    questao_3()
    questao_4()
    questao_6("Lenna.png")
    questao_7()
    questao_8()
