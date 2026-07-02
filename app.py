"""
Demo funcional - Comparacion genomica de biodiversidad peruana (Grupo A)
Analisis y Diseno de Algoritmos - Universidad ESAN

Reutiliza el MISMO algoritmo del main.py original (Needleman-Wunsch clasico +
MST con Kruskal) y le agrega una interfaz web con Streamlit para cumplir con
el entregable de "Demo funcional": mostrar entradas, salidas y metricas en vivo.
"""

import glob
import heapq
import io
import os
import time

import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------
# Configuracion de puntajes (identica al main.py original)
# ----------------------------------------------------------------------------
MATCH = 1
MISMATCH = -1
GAP = -2
CARPETA_DATA = "data"

st.set_page_config(
    page_title="Comparacion Genomica - Camelidos Peruanos",
    layout="wide",
)


# ----------------------------------------------------------------------------
# ALGORITMO (mismas funciones que main.py, sin modificar la logica)
# ----------------------------------------------------------------------------
def leer_fasta_desde_texto(contenido, limite):
    """Extrae la secuencia de un archivo FASTA ya leido como texto."""
    secuencia = ""
    for linea in contenido.splitlines():
        linea = linea.strip().upper()
        if linea and not linea.startswith(">"):
            secuencia += linea
    return secuencia[:limite]


def needleman_wunsch(seq1, seq2):
    """Construye la matriz de puntajes y la matriz del camino (traceback)."""
    filas = len(seq1) + 1
    columnas = len(seq2) + 1

    matriz = [[0 for _ in range(columnas)] for _ in range(filas)]
    camino = [["" for _ in range(columnas)] for _ in range(filas)]

    for i in range(1, filas):
        matriz[i][0] = i * GAP
        camino[i][0] = "arriba"

    for j in range(1, columnas):
        matriz[0][j] = j * GAP
        camino[0][j] = "izquierda"

    for i in range(1, filas):
        for j in range(1, columnas):
            if seq1[i - 1] == seq2[j - 1]:
                puntaje_letras = MATCH
            else:
                puntaje_letras = MISMATCH

            diagonal = matriz[i - 1][j - 1] + puntaje_letras
            arriba = matriz[i - 1][j] + GAP
            izquierda = matriz[i][j - 1] + GAP

            mejor = max(diagonal, arriba, izquierda)
            matriz[i][j] = mejor

            if mejor == diagonal:
                camino[i][j] = "diagonal"
            elif mejor == arriba:
                camino[i][j] = "arriba"
            else:
                camino[i][j] = "izquierda"

    return matriz, camino


def reconstruir_alineamiento(seq1, seq2, camino):
    """Retrocede por la matriz para obtener el alineamiento final."""
    i = len(seq1)
    j = len(seq2)

    alineada1 = []
    alineada2 = []
    ruta = [(i, j)]

    while i > 0 or j > 0:
        direccion = camino[i][j]

        if i > 0 and j > 0 and direccion == "diagonal":
            alineada1.append(seq1[i - 1])
            alineada2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or direccion == "arriba"):
            alineada1.append(seq1[i - 1])
            alineada2.append("-")
            i -= 1
        else:
            alineada1.append("-")
            alineada2.append(seq2[j - 1])
            j -= 1

        ruta.append((i, j))

    alineada1.reverse()
    alineada2.reverse()
    ruta.reverse()

    return "".join(alineada1), "".join(alineada2), ruta


def calcular_metricas(alineada1, alineada2):
    coincidencias = 0
    diferencias = 0
    gaps = 0

    for letra1, letra2 in zip(alineada1, alineada2):
        if letra1 == "-" or letra2 == "-":
            gaps += 1
        elif letra1 == letra2:
            coincidencias += 1
        else:
            diferencias += 1

    longitud = len(alineada1)
    identidad = coincidencias / longitud
    distancia = 1 - identidad

    return coincidencias, diferencias, gaps, longitud, identidad, distancia


def kruskal(especies, aristas):
    """Construye el MST seleccionando primero las distancias menores."""
    padre = {especie: especie for especie in especies}

    def buscar_raiz(especie):
        while padre[especie] != especie:
            especie = padre[especie]
        return especie

    heap = aristas.copy()
    heapq.heapify(heap)
    mst = []

    while heap and len(mst) < len(especies) - 1:
        distancia, especie1, especie2 = heapq.heappop(heap)
        raiz1 = buscar_raiz(especie1)
        raiz2 = buscar_raiz(especie2)
        if raiz1 != raiz2:
            padre[raiz2] = raiz1
            mst.append((especie1, especie2, distancia))

    return mst


# ----------------------------------------------------------------------------
# GRAFICOS
# ----------------------------------------------------------------------------
def figura_matriz(matriz, ruta, especie1, especie2):
    x_ruta = [columna for _fila, columna in ruta]
    y_ruta = [fila for fila, _columna in ruta]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    imagen = ax.imshow(matriz, origin="upper", aspect="auto")
    ax.plot(x_ruta, y_ruta, linewidth=1, color="red")
    ax.set_title(f"Matriz Needleman-Wunsch y traceback\n{especie1} vs {especie2}")
    ax.set_xlabel("Secuencia 2")
    ax.set_ylabel("Secuencia 1")
    fig.colorbar(imagen, ax=ax, label="Puntaje")
    fig.tight_layout()
    return fig


def figura_mst(mst, especies):
    import math

    n = len(especies)
    posiciones = {}
    for idx, especie in enumerate(especies):
        angulo = 2 * math.pi * idx / n
        posiciones[especie] = (math.cos(angulo) * 3, math.sin(angulo) * 3)

    fig, ax = plt.subplots(figsize=(6, 5))

    for especie1, especie2, distancia in mst:
        x1, y1 = posiciones[especie1]
        x2, y2 = posiciones[especie2]
        ax.plot([x1, x2], [y1, y2], linewidth=2)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{distancia:.4f}", fontsize=9)

    for especie, (x, y) in posiciones.items():
        ax.scatter(x, y, s=2200)
        ax.text(x, y, especie, ha="center", va="center", fontsize=10)

    ax.set_title("MST de cercania genetica")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# CARGA DE SECUENCIAS (archivos por defecto en data/ o subidos por el usuario)
# ----------------------------------------------------------------------------
def cargar_especies_por_defecto():
    especies = {}
    for ruta in sorted(glob.glob(os.path.join(CARPETA_DATA, "*.fasta"))):
        nombre = os.path.splitext(os.path.basename(ruta))[0].capitalize()
        with open(ruta, "r", encoding="utf-8") as archivo:
            especies[nombre] = archivo.read()
    return especies


# ----------------------------------------------------------------------------
# INTERFAZ (Streamlit)
# ----------------------------------------------------------------------------
st.title("🧬 Comparacion genomica de biodiversidad peruana")
st.caption(
    "Grupo A - Programacion Dinamica | Needleman-Wunsch + MST (Kruskal) | "
    "Analisis y Diseno de Algoritmos - Universidad ESAN"
)

with st.sidebar:
    st.header("1. Datos de entrada")

    especies_defecto = cargar_especies_por_defecto()
    usar_defecto = st.checkbox(
        f"Usar archivos de ejemplo ({len(especies_defecto)} especies en /data)",
        value=bool(especies_defecto),
    )

    archivos_subidos = None
    if not usar_defecto:
        archivos_subidos = st.file_uploader(
            "Sube archivos FASTA (uno por especie)",
            type=["fasta", "fa", "txt"],
            accept_multiple_files=True,
        )

    st.header("2. Parametros del algoritmo")
    limite = st.slider(
        "Nucleotidos a usar por secuencia",
        min_value=50,
        max_value=2000,
        value=300,
        step=50,
        help=(
            "Needleman-Wunsch clasico es O(n*m) en tiempo y memoria. "
            "Valores altos son mas precisos pero mas lentos en la demo en vivo."
        ),
    )

    st.caption(f"Puntajes fijos: match = {MATCH}, mismatch = {MISMATCH}, gap = {GAP}")

    ejecutar = st.button("▶ Ejecutar analisis", type="primary", use_container_width=True)

# Construir diccionario de especies segun la fuente elegida
secuencias_crudas = {}
if usar_defecto:
    secuencias_crudas = especies_defecto
elif archivos_subidos:
    for archivo in archivos_subidos:
        nombre = os.path.splitext(archivo.name)[0].capitalize()
        secuencias_crudas[nombre] = archivo.read().decode("utf-8", errors="ignore")

if not secuencias_crudas:
    st.info("⬅ Selecciona los archivos de ejemplo o sube tus propios FASTA para comenzar.")
    st.stop()

especies_disponibles = list(secuencias_crudas.keys())
st.write(f"**Especies cargadas:** {', '.join(especies_disponibles)}")

if len(especies_disponibles) < 2:
    st.warning("Se necesitan al menos 2 especies para comparar.")
    st.stop()

if not ejecutar:
    st.info("Ajusta los parametros en la barra lateral y presiona **Ejecutar analisis**.")
    st.stop()

# ----------------------------------------------------------------------------
# EJECUCION DEL ANALISIS (entradas -> salidas -> metricas, en vivo)
# ----------------------------------------------------------------------------
secuencias = {
    especie: leer_fasta_desde_texto(texto, limite)
    for especie, texto in secuencias_crudas.items()
}

with st.expander("Ver longitud de las secuencias leidas"):
    for especie, seq in secuencias.items():
        st.write(f"- **{especie}**: {len(seq)} nucleotidos")

resultados = []
aristas = []
alineamientos = {}
figuras_matriz = {}

progreso = st.progress(0.0, text="Procesando comparaciones...")
pares = [
    (especies_disponibles[i], especies_disponibles[j])
    for i in range(len(especies_disponibles))
    for j in range(i + 1, len(especies_disponibles))
]

for indice, (especie1, especie2) in enumerate(pares):
    inicio = time.time()
    seq1 = secuencias[especie1]
    seq2 = secuencias[especie2]

    matriz, camino = needleman_wunsch(seq1, seq2)
    puntaje = matriz[-1][-1]

    alineada1, alineada2, ruta = reconstruir_alineamiento(seq1, seq2, camino)
    coincidencias, diferencias, gaps, longitud, identidad, distancia = calcular_metricas(
        alineada1, alineada2
    )
    segundos = time.time() - inicio

    resultados.append(
        {
            "Especie 1": especie1,
            "Especie 2": especie2,
            "Puntaje NW": puntaje,
            "Coincidencias": coincidencias,
            "Diferencias": diferencias,
            "Gaps": gaps,
            "Longitud alineada": longitud,
            "Identidad (%)": round(identidad * 100, 4),
            "Distancia": round(distancia, 6),
            "Tiempo (s)": round(segundos, 2),
        }
    )
    aristas.append((distancia, especie1, especie2))
    alineamientos[(especie1, especie2)] = (alineada1, alineada2, puntaje)

    # La primera comparacion se guarda como figura (igual que main.py)
    if indice == 0:
        figuras_matriz[(especie1, especie2)] = figura_matriz(matriz, ruta, especie1, especie2)

    progreso.progress((indice + 1) / len(pares), text=f"{especie1} vs {especie2} listo")

progreso.empty()

mst = kruskal(especies_disponibles, aristas)

# ----------------------------------------------------------------------------
# SALIDAS EN PANTALLA
# ----------------------------------------------------------------------------
tab_resumen, tab_alineamiento, tab_matriz, tab_mst = st.tabs(
    ["📊 Resultados", "🧬 Alineamiento detallado", "🗺️ Matriz + traceback", "🌳 Arbol MST"]
)

with tab_resumen:
    st.subheader("Tabla de resultados (Needleman-Wunsch)")
    st.dataframe(resultados, use_container_width=True)

    csv_buffer = io.StringIO()
    import csv as csv_module

    escritor = csv_module.DictWriter(csv_buffer, fieldnames=list(resultados[0].keys()))
    escritor.writeheader()
    escritor.writerows(resultados)
    st.download_button(
        "⬇ Descargar resultados_nw.csv",
        data=csv_buffer.getvalue(),
        file_name="resultados_nw.csv",
        mime="text/csv",
    )

with tab_alineamiento:
    st.subheader("Ver un alineamiento par a par")
    opcion = st.selectbox(
        "Selecciona la pareja de especies",
        options=list(alineamientos.keys()),
        format_func=lambda par: f"{par[0]} vs {par[1]}",
    )
    alineada1, alineada2, puntaje = alineamientos[opcion]
    st.write(f"**Puntaje NW:** {puntaje}")

    bloque = ""
    for i in range(0, len(alineada1), 80):
        bloque += alineada1[i : i + 80] + "\n"
        bloque += alineada2[i : i + 80] + "\n\n"
    st.text_area("Alineamiento (bloques de 80 caracteres)", bloque, height=300)

    st.download_button(
        "⬇ Descargar este alineamiento (.txt)",
        data=bloque,
        file_name=f"alineamiento_{opcion[0]}_vs_{opcion[1]}.txt",
    )

with tab_matriz:
    st.subheader("Matriz de puntajes y camino de traceback")
    if figuras_matriz:
        par_guardado = next(iter(figuras_matriz))
        st.pyplot(figuras_matriz[par_guardado])
        st.caption(
            f"Se muestra la primera comparacion procesada ({par_guardado[0]} vs "
            f"{par_guardado[1]}), igual que en el script original."
        )
    else:
        st.write("No hay figura disponible.")

with tab_mst:
    st.subheader("Arbol de expansion minima (Kruskal)")
    st.dataframe(
        [{"Especie 1": e1, "Especie 2": e2, "Distancia": round(d, 6)} for e1, e2, d in mst],
        use_container_width=True,
    )
    st.pyplot(figura_mst(mst, especies_disponibles))

st.success("Proceso terminado ✅")
