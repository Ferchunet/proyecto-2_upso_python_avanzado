import os
from urllib.request import urlopen

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

# Carpeta donde se guardan los resultados.
CARPETA_RESULTADOS = "fotos_mejoradas"
# Carpeta donde se buscan las fotos cuando el usuario escribe solo el nombre.
CARPETA_IMAGENES = "."

# Ancho y alto de cada red. El orden es (ancho, alto), igual que img.size.
TAMANOS = {
    "Youtube": (1280, 720),
    "Instagram": (1080, 1080),
    "Twitter": (1600, 900),
    "Facebook": (1200, 630),
}

# Los 9 filtros de la clase. La clave es el nombre que elige el usuario.
FILTROS = {
    "Blur": ImageFilter.BLUR,
    "Contour": ImageFilter.CONTOUR,
    "Detail": ImageFilter.DETAIL,
    "Edge Enhance": ImageFilter.EDGE_ENHANCE,
    "Edge Enhance More": ImageFilter.EDGE_ENHANCE_MORE,
    "Emboss": ImageFilter.EMBOSS,
    "Find Edges": ImageFilter.FIND_EDGES,
    "Sharpen": ImageFilter.SHARPEN,
    "Smooth": ImageFilter.SMOOTH,
}


def obtener_ruta(nombre):
    # Si ya es una ruta o un link, la deja. Si es solo el nombre, lo busca en la carpeta.
    nombre = nombre.strip().strip("/\\")
    if nombre.startswith("http") or os.path.isabs(nombre):
        return nombre
    return os.path.join(CARPETA_IMAGENES, nombre)


def abrir_imagen(ruta):
    # Drive o URL
    if ruta.startswith("http"):
        return Image.open(urlopen(ruta))
    if not os.path.exists(ruta):
        raise FileNotFoundError("No encuentro la imagen en esa ruta")
    return Image.open(ruta)


def como_imagen(img):
    # Si llega una ruta, la abre. Si ya es una imagen de Pillow, la usa.
    if img is None:
        raise ValueError("Primero tenes que abrir la foto con la opcion 1.")
    if isinstance(img, str):
        return abrir_imagen(img).convert("RGB")
    return img.convert("RGB")


def mostrar(imagenes, titulos=None, tam=4, archivo=None):
    # Muestra una o varias imágenes una al lado de la otra
    if not isinstance(imagenes, (list, tuple)):
        imagenes = [imagenes]
    if titulos is None:
        titulos = [""] * len(imagenes)
    if isinstance(titulos, str):
        titulos = [titulos]

    fig, ejes = plt.subplots(1, len(imagenes), figsize=(tam * len(imagenes), tam))
    ejes = np.atleast_1d(ejes)
    for eje, img, titulo in zip(ejes, imagenes, titulos):
        img = np.asarray(img)                               # por si es una imagen de Pillow
        if img.ndim == 2 and img.dtype == np.uint8:
            eje.imshow(img, cmap="gray", vmin=0, vmax=255)  # gris: 0 = negro, 255 = blanco
        else:
            eje.imshow(img, cmap="gray")                    # cmap solo se usa si la imagen es 2D
        eje.set_title(titulo)
        eje.axis("off")

    # Guarda la figura cuando el punto 2 o el 3 pasan un nombre de archivo.
    destino = None
    if archivo:
        os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
        destino = os.path.join(CARPETA_RESULTADOS, archivo)
        fig.savefig(destino)
    plt.show()
    plt.close(fig)
    return destino


def redimensionar_para_red(ruta, red):
    # Punto 1. ruta y red son dos strings.
    red = red.strip().capitalize()
    if red not in TAMANOS:
        raise ValueError("La red tiene que ser Youtube, Instagram, Twitter o Facebook")

    img = abrir_imagen(ruta).convert("RGB")
    ancho, alto = img.size
    ancho_obj, alto_obj = TAMANOS[red]

    # La escala mas grande cubre el marco. Los dos lados se multiplican por el mismo numero.
    factor = max(ancho_obj / ancho, alto_obj / alto)
    nuevo_ancho = int(ancho * factor)
    nuevo_alto = int(alto * factor)
    if nuevo_ancho < ancho_obj:
        nuevo_ancho = ancho_obj
    if nuevo_alto < alto_obj:
        nuevo_alto = alto_obj

    img_escalada = img.resize((nuevo_ancho, nuevo_alto))

    # Recorte centrado: (izquierda, arriba, derecha, abajo)
    izquierda = (nuevo_ancho - ancho_obj) // 2
    arriba = (nuevo_alto - alto_obj) // 2
    derecha = izquierda + ancho_obj
    abajo = arriba + alto_obj
    img_final = img_escalada.crop((izquierda, arriba, derecha, abajo))

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    nombre = os.path.splitext(os.path.basename(ruta.split("?")[0]))[0]
    destino = os.path.join(CARPETA_RESULTADOS, nombre + "_" + red + ".jpg")
    img_final.save(destino)
    return img_final, destino


def ecualizar_canal(canal):
    # Cuenta los brillos, acumula y cambia cada pixel por el valor nuevo.
    numeros = np.array(canal)
    conteo, _ = np.histogram(numeros, bins=256, range=(0, 256))
    probabilidad = conteo / numeros.size
    acumulada = np.cumsum(probabilidad)
    tabla = np.round(acumulada * 255).astype("uint8")
    return Image.fromarray(tabla[numeros])


def ajustar_contraste(img):
    # Punto 2. Separa el claro del oscuro en cada color y los vuelve a unir.
    img = como_imagen(img)
    rojo, verde, azul = img.split()
    return Image.merge("RGB", (
        ecualizar_canal(rojo),
        ecualizar_canal(verde),
        ecualizar_canal(azul),
    ))


def mostrar_contraste(img, nombre=None):
    # Muestra la original y la ecualizada juntas y guarda la figura.
    if isinstance(img, str) and nombre is None:
        nombre = os.path.splitext(os.path.basename(img))[0]
    if nombre is None:
        nombre = "foto"
    original = como_imagen(img)
    mejorada = ajustar_contraste(original)
    destino = mostrar(
        [original, mejorada],
        ["Original", "Ecualizada"],
        archivo=nombre + "_contraste.png",
    )
    return mejorada, destino


def aplicar_filtro(img, nombre, nombre_archivo=None):
    clave = " ".join(nombre.strip().split())
    nombre_filtro = None
    filtro = None
    for candidato, valor in FILTROS.items():
        if candidato.lower() == clave.lower():
            nombre_filtro = candidato
            filtro = valor
            break
    if filtro is None:
        raise ValueError("El filtro tiene que ser: " + ", ".join(FILTROS))

    original = como_imagen(img)
    filtrada = original.filter(filtro)
    if nombre_archivo is None:
        nombre_archivo = "filtro"
    destino = mostrar(
        [original, filtrada],
        ["Original", nombre_filtro],
        archivo=nombre_archivo + "_" + nombre_filtro.replace(" ", "_") + ".png",
    )
    return filtrada, destino


def detectar_bordes_canny(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.Canny(img, 50, 100)
  return img


def detectar_bordes_laplaciano(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.Laplacian(img, cv2.CV_64F)
  img = cv2.convertScaleAbs(img)
  return img


def main():
    # Para contraste, filtros y bordes hay que pasar antes por la opcion 1.
    img_actual = None
    nombre_base = None

    while True:
        print()
        print("1. Abrir y redimensionar para una red")
        print("2. Ajustar contraste")
        print("3. Aplicar un filtro")
        print("4. Bordes y contornos")
        print("5. Salir")
        opcion = input("Opcion: ").strip()

        if opcion == "5":
            print("Listo.")
            break

        try:
            if opcion == "1":
                entrada = input("Nombre o ruta de la foto: ")
                ruta = obtener_ruta(entrada)
                print("Redes:")
                redes = list(TAMANOS)
                for i, red in enumerate(redes, start=1):
                    print(str(i) + ". " + red)
                entrada = input("Numero o nombre de la red: ").strip()
                if entrada.isdigit():
                    numero = int(entrada)
                    if not 1 <= numero <= len(redes):
                        raise ValueError("La entrada esta fuera del rango de opciones proveidas.")
                    red = redes[numero - 1]
                else:
                    red = entrada
                img_actual, destino = redimensionar_para_red(ruta, red)
                nombre_base = os.path.splitext(os.path.basename(ruta.split("?")[0]))[0]
                print("Tamaño nuevo:", img_actual.size)
                print("Guardada en:", destino)
                continue

            if img_actual is None:
                raise ValueError("Primero tenes que abrir la foto con la opcion 1.")

            if opcion == "2":
                mejorada, destino = mostrar_contraste(img_actual, nombre_base)
                print("Figura guardada en:", destino)
                continue

            if opcion == "3":
                print("Filtros:")
                nombres = list(FILTROS)
                for i, nombre_filtro in enumerate(nombres, start=1):
                    print(str(i) + ". " + nombre_filtro)
                entrada = input("Numero del filtro: ").strip()
                if not entrada.isdigit():
                    raise ValueError("La entrada no es un numero valido.")
                numero = int(entrada)
                if not 1 <= numero <= len(nombres):
                    raise ValueError("La entrada esta fuera del rango de opciones proveidas.")
                nombre_filtro = nombres[numero - 1]
                filtrada, destino = aplicar_filtro(img_actual, nombre_filtro, nombre_base)
                print("Figura guardada en:", destino)
                continue

            if opcion == "4":
                # La convertimos a RGB porque matplotlib asume que la imagen es en RGB
                img = np.array(img_actual)
                img_canny = detectar_bordes_canny(img)
                os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
                destino_canny = os.path.join(CARPETA_RESULTADOS, nombre_base + "_Bordes_Canny.png")
                cv2.imwrite(destino_canny, img_canny)
                mostrar([img, img_canny], ["Original", "Bordes Canny"])

                img_laplaciano = detectar_bordes_laplaciano(img)
                destino_laplaciano = os.path.join(CARPETA_RESULTADOS, nombre_base + "_Bordes_Laplaciano.png")
                cv2.imwrite(destino_laplaciano, img_laplaciano)
                mostrar([img, img_laplaciano], ["Original", "Bordes Laplaciano"])
                print("Figuras guardadas en:", destino_canny, "y", destino_laplaciano)
                continue

            print("Esa opcion no esta. Elegi 1, 2, 3, 4 o 5.")
        except (ValueError, OSError) as error:
            print(error)


if __name__ == "__main__":
    main()
