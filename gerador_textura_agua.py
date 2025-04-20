import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
from PIL import Image, ImageTk
import random
import time
import traceback
import math

# Import Numba e CUDA
try:
    from numba import cuda, jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    class cuda:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def jit(func): return func # Retorna a função original
        @staticmethod
        def to_device(arr): return arr # Placeholder
        @staticmethod
        def device_array(shape, dtype): return np.zeros(shape, dtype=dtype) # Placeholder
        @staticmethod
        def synchronize(): pass # Placeholder
    def jit(options={}):
        def decorator(func):
             return func
        return decorator
    njit = jit
    prange = range

# ======================================================
# Funções e Kernel CUDA para Geração de Ruído Perlin
# (Reutilizadas do gerador de heightmap, podem precisar de adaptação)
# ======================================================

# --- Funções Auxiliares (Device Functions) ---
@cuda.jit(device=True)
def fade(t):
    """Função de interpolação suave (6t^5 - 15t^4 + 10t^3)."""
    return t * t * t * (t * (t * 6 - 15) + 10)

@cuda.jit(device=True)
def lerp(t, a, b):
    """Interpolação linear."""
    return a + t * (b - a)

# Adiciona clamp como função device
@cuda.jit(device=True)
def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))

@cuda.jit(device=True)
def grad(hash_val, x, y):
    """Calcula o produto escalar entre vetor gradiente pseudo-aleatório e vetor distância."""
    h = hash_val & 7 # Pega os últimos 3 bits (0-7)
    if h == 0: return x + y
    if h == 1: return -x + y
    if h == 2: return x - y
    if h == 3: return -x - y
    if h == 4: return x + x + y # Gradientes não normalizados para simplificar
    if h == 5: return -x - x + y
    if h == 6: return x + y + y
    if h == 7: return -x - y - y
    return 0 # Caso improvável

@cuda.jit(device=True)
def perlin_noise_2d(x, y, perm_table):
    """Calcula o ruído Perlin 2D para um ponto (x, y) usando uma tabela de permutação."""
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    u = fade(xf)
    v = fade(yf)
    p = perm_table
    aa = p[p[xi ] + yi ]
    ab = p[p[xi ] + yi + 1]
    ba = p[p[xi + 1] + yi ]
    bb = p[p[xi + 1] + yi + 1]
    g_aa = grad(aa, xf, yf)
    g_ba = grad(ba, xf - 1, yf)
    g_ab = grad(ab, xf, yf - 1)
    g_bb = grad(bb, xf - 1, yf - 1)
    x1 = lerp(u, g_aa, g_ba)
    x2 = lerp(u, g_ab, g_bb)
    return lerp(v, x1, x2)

# --- Kernel CUDA Principal (Adaptado para gerar apenas ruído base por enquanto) ---
@cuda.jit
def noise_kernel_gpu(output_array, width, height, scale, octaves, persistence, lacunarity, seed, perm_table):
    """Kernel CUDA para gerar ruído base com oitavas."""
    idx, idy = cuda.grid(2)

    if idx >= width or idy >= height:
        return

    nx = float(idx) / width
    ny = float(idy) / height

    total_noise = 0.0
    frequency = 1.0
    amplitude = 1.0
    effective_scale = scale # Ajustar escala conforme necessidade

    for k in range(octaves):
        noise_val = perlin_noise_2d(nx * effective_scale * frequency,
                                      ny * effective_scale * frequency,
                                      perm_table)
        total_noise += noise_val * amplitude
        amplitude *= persistence
        frequency *= lacunarity

    output_array[idy, idx] = total_noise

# Adiciona kernel para normalização [0, 1]
@cuda.jit
def normalize_kernel_gpu(input_map, output_map, width, height, min_val, range_val):
    idx, idy = cuda.grid(2)
    if idx >= width or idy >= height:
        return
    if range_val > 1e-6:
        output_map[idy, idx] = (input_map[idy, idx] - min_val) / range_val
    else:
        output_map[idy, idx] = 0.5 # Valor neutro se não houver variação

# --- Kernel CUDA para Combinar Ruídos ---
@cuda.jit
def combine_noise_kernel_gpu(base_noise_map, detail_noise_map, combined_map_out, width, height, detail_influence):
    """Combina dois mapas de ruído na GPU (normalizados para [-1, 1] antes)."""
    idx, idy = cuda.grid(2)
    if idx >= width or idy >= height:
        return

    # Assume que base_noise_map e detail_noise_map já estão em [-1, 1] ou similar
    base_val = base_noise_map[idy, idx]
    detail_val = detail_noise_map[idy, idx]

    combined_map_out[idy, idx] = base_val * (1.0 - detail_influence) + detail_val * detail_influence


# --- Kernel CUDA para Calcular Mapa de Normais ---
@cuda.jit
def normal_map_kernel_gpu(displacement_map, normal_map_out, width, height):
    """Calcula o mapa de normais na GPU a partir do mapa de deslocamento."""
    idx, idy = cuda.grid(2)

    if idx >= width or idy >= height:
        return

    # Calcula gradientes usando diferenças finitas (central se possível)
    # Acessa pixels vizinhos, cuidado com as bordas!
    h_center = displacement_map[idy, idx]

    # Amostra vizinhos com clamp nas bordas
    h_left = displacement_map[idy, clamp(idx - 1, 0, width - 1)]
    h_right = displacement_map[idy, clamp(idx + 1, 0, width - 1)]
    h_down = displacement_map[clamp(idy - 1, 0, height - 1), idx]
    h_up = displacement_map[clamp(idy + 1, 0, height - 1), idx]

    # Calcula derivadas parciais (diferença dividida por distância, aqui normalizada a 2 pixels)
    dz_dx = (h_right - h_left) * 0.5 # Multiplicador 0.5 para normalizar a distância
    dz_dy = (h_up - h_down) * 0.5

    # Calcula vetor normal não normalizado (-dz/dx, -dz/dy, 1)
    # O fator de escala pode ser ajustado para controlar a "força" do normal map
    strength = 1.0 # Pode ser um parâmetro
    nx = -dz_dx * strength
    ny = -dz_dy * strength
    nz = 1.0

    # Normaliza o vetor
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm > 1e-6: # Evita divisão por zero
        inv_norm = 1.0 / norm
        nx *= inv_norm
        ny *= inv_norm
        nz *= inv_norm
    else:
        # Se a norma for zero (plano), aponta para cima
        nx = 0.0
        ny = 0.0
        nz = 1.0

    # Mapeia componentes normalizados [-1, 1] para espaço de cor [0, 255]
    # Normal X -> R
    normal_map_out[idy, idx, 0] = clamp(int((nx + 1.0) * 0.5 * 255.0), 0, 255)
    # Normal Y -> G (Invertido para convenção comum)
    normal_map_out[idy, idx, 1] = clamp(int((-ny + 1.0) * 0.5 * 255.0), 0, 255)
    # Normal Z -> B (Mapeado de [0, 1] para [0, 255], pois Z normalizado é sempre >= 0)
    normal_map_out[idy, idx, 2] = clamp(int(nz * 255.0), 0, 255)

# --- Kernel CUDA para Calcular Mapa de Rugosidade ---
@cuda.jit
def roughness_map_kernel_gpu(noise_normalized_01, roughness_map_out, width, height, foam_level, base_roughness, foam_roughness):
    """Calcula o mapa de rugosidade na GPU."""
    idx, idy = cuda.grid(2)
    if idx >= width or idy >= height:
        return

    noise_val = noise_normalized_01[idy, idx]

    # Determina se é espuma (0.0 ou 1.0)
    is_foam = 1.0 if noise_val > foam_level else 0.0

    # Interpola entre rugosidade base e rugosidade da espuma
    roughness = lerp(is_foam, base_roughness, foam_roughness)

    # Mapeia para [0, 255] (formato L uint8)
    roughness_map_out[idy, idx] = clamp(int(roughness * 255.0), 0, 255)


# --- Kernel CUDA para Calcular Mapa de Cor ---
@cuda.jit
def color_map_kernel_gpu(noise_normalized_01, color_map_out, width, height,
                         base_r, base_g, base_b, depth_r, depth_g, depth_b,
                         foam_thresh, foam_intensity):
    """Calcula o mapa de cores na GPU."""
    idx, idy = cuda.grid(2)
    if idx >= width or idy >= height:
        return

    noise_val = noise_normalized_01[idy, idx]

    # Interpola entre cor profunda e cor base (valores 0-255)
    water_r = lerp(noise_val, float(depth_r), float(base_r))
    water_g = lerp(noise_val, float(depth_g), float(base_g))
    water_b = lerp(noise_val, float(depth_b), float(base_b))

    # Adiciona espuma (interpola para branco)
    is_foam = 1.0 if noise_val > foam_thresh else 0.0
    foam_factor = is_foam * foam_intensity

    final_r = lerp(foam_factor, water_r, 255.0)
    final_g = lerp(foam_factor, water_g, 255.0)
    final_b = lerp(foam_factor, water_b, 255.0)

    # Salva como RGB uint8
    color_map_out[idy, idx, 0] = clamp(int(final_r), 0, 255)
    color_map_out[idy, idx, 1] = clamp(int(final_g), 0, 255)
    color_map_out[idy, idx, 2] = clamp(int(final_b), 0, 255)

# ======================================================
# Funções Numba @njit para Geração de Ruído Perlin (CPU)
# (Reutilizadas e adaptadas)
# ======================================================
@njit
def fade_cpu(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@njit
def lerp_cpu(t, a, b):
    return a + t * (b - a)

@njit
def grad_cpu(hash_val, x, y):
    h = hash_val & 7
    if h == 0: return x + y
    if h == 1: return -x + y
    if h == 2: return x - y
    if h == 3: return -x - y
    if h == 4: return x + x + y
    if h == 5: return -x - x + y
    if h == 6: return x + y + y
    if h == 7: return -x - y - y
    return 0

@njit
def perlin_noise_2d_cpu(x, y, perm_table):
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    u = fade_cpu(xf)
    v = fade_cpu(yf)
    p = perm_table
    aa = p[p[xi ] + yi ]
    ab = p[p[xi ] + yi + 1]
    ba = p[p[xi + 1] + yi ]
    bb = p[p[xi + 1] + yi + 1]
    g_aa = grad_cpu(aa, xf, yf)
    g_ba = grad_cpu(ba, xf - 1, yf)
    g_ab = grad_cpu(ab, xf, yf - 1)
    g_bb = grad_cpu(bb, xf - 1, yf - 1)
    x1 = lerp_cpu(u, g_aa, g_ba)
    x2 = lerp_cpu(u, g_ab, g_bb)
    return lerp_cpu(v, x1, x2)

@njit(parallel=True)
def noise_loop_cpu(width, height, scale, octaves, persistence, lacunarity, seed, perm_table):
    """Loop principal para gerar ruído base na CPU com Numba (paralelizado)."""
    output_array = np.zeros((height, width), dtype=np.float64)

    for i in prange(height):
        for j in range(width):
            nx = float(j) / width
            ny = float(i) / height

            total_noise = 0.0
            frequency = 1.0
            amplitude = 1.0
            effective_scale = scale

            for k in range(octaves):
                noise_val = perlin_noise_2d_cpu(nx * effective_scale * frequency,
                                                ny * effective_scale * frequency,
                                                perm_table)
                total_noise += noise_val * amplitude
                amplitude *= persistence
                frequency *= lacunarity
            output_array[i, j] = total_noise

    return output_array

# ======================================================
# Classe Principal da Aplicação Tkinter
# ======================================================

class WaterTextureGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gerador de Textura de Água")
        self.root.geometry("1350x800") # Aumentar tamanho para mais controles/previews

        style = ttk.Style()
        style.theme_use('clam')

        # --- Frame Principal Esquerdo (Scrollable) ---
        left_frame_container = ttk.Frame(root)
        left_frame_container.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 5), pady=5)

        canvas = tk.Canvas(left_frame_container, borderwidth=0, width=400) # Definir largura inicial
        self.scrollable_frame = ttk.Frame(canvas, padding="10")
        scrollbar = ttk.Scrollbar(left_frame_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas_frame_id = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_frame_id, width=event.width) # Ajusta largura do frame interno

        self.scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_frame_id, width=e.width)) # Ajusta na redimensão do canvas

        def _on_mousewheel(event):
            # Ajustar delta para diferentes OS
            delta = 0
            if event.num == 4: delta = -1 # Linux scroll up
            elif event.num == 5: delta = 1 # Linux scroll down
            elif hasattr(event, 'delta'): delta = -int(event.delta / 120) # Windows/macOS
            if delta != 0:
                canvas.yview_scroll(delta, "units")

        # Bind universal (funciona melhor com bind_all no root ou container principal)
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)


        # --- Painel Direito para Imagens ---
        right_panel = ttk.Frame(root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ======================================================================
        # Adicionar Controles ao self.scrollable_frame
        # ======================================================================

        # --- Frame de Configurações Gerais ---
        settings_frame = ttk.LabelFrame(self.scrollable_frame, text="Configurações Gerais", padding="10")
        settings_frame.pack(fill=tk.X, pady=5, anchor='n')
        ttk.Label(settings_frame, text="Largura:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.width_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.width_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(settings_frame, text="Altura:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.height_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.height_var, width=8).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(settings_frame, text="Semente:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.seed_var = tk.IntVar(value=random.randint(0, 1000))
        self.seed_entry = ttk.Entry(settings_frame, textvariable=self.seed_var, width=8)
        self.seed_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Aleatória", command=self.randomize_seed, width=8).grid(row=1, column=2, padx=5, pady=5)

        # --- Frame de Processamento ---
        self.proc_frame = ttk.LabelFrame(self.scrollable_frame, text="Processamento", padding="10")
        self.proc_frame.pack(fill=tk.X, pady=5, anchor='n')
        self.processing_device = tk.StringVar(value="CPU")
        cpu_radio = ttk.Radiobutton(self.proc_frame, text="CPU", variable=self.processing_device, value="CPU", command=self._check_gpu_availability)
        cpu_radio.pack(side=tk.LEFT, padx=5)
        gpu_radio = ttk.Radiobutton(self.proc_frame, text="GPU (CUDA)", variable=self.processing_device, value="GPU", command=self._check_gpu_availability)
        gpu_radio.pack(side=tk.LEFT, padx=5)
        self.gpu_status_label = ttk.Label(self.proc_frame, text="")
        self.gpu_status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # --- Frame de Parâmetros das Ondas (Ruído Base) ---
        wave_noise_frame = ttk.LabelFrame(self.scrollable_frame, text="Parâmetros das Ondas (Ruído Base)", padding="10")
        wave_noise_frame.pack(fill=tk.X, pady=5, anchor='n')
        # (Adicionaremos sliders para escala, oitavas, etc. aqui)
        ttk.Label(wave_noise_frame, text="Escala:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.scale_var = tk.DoubleVar(value=25.0)
        self.scale_display_var = tk.StringVar()
        ttk.Scale(wave_noise_frame, from_=1.0, to=100.0, orient=tk.HORIZONTAL, variable=self.scale_var, length=180, command=self._update_noise_labels).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(wave_noise_frame, textvariable=self.scale_display_var, width=5).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(wave_noise_frame, text="Oitavas:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.octaves_var = tk.IntVar(value=8)
        ttk.Scale(wave_noise_frame, from_=1, to=16, orient=tk.HORIZONTAL, variable=self.octaves_var, command=lambda v: self.octaves_var.set(int(float(v)))).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(wave_noise_frame, textvariable=self.octaves_var, width=5).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(wave_noise_frame, text="Persist.:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.persistence_var = tk.DoubleVar(value=0.5)
        self.persistence_display_var = tk.StringVar()
        ttk.Scale(wave_noise_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.persistence_var, command=self._update_noise_labels).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(wave_noise_frame, textvariable=self.persistence_display_var, width=5).grid(row=2, column=2, padx=5, pady=2)

        ttk.Label(wave_noise_frame, text="Lacunar.:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.lacunarity_var = tk.DoubleVar(value=2.0)
        self.lacunarity_display_var = tk.StringVar()
        ttk.Scale(wave_noise_frame, from_=1.0, to=4.0, orient=tk.HORIZONTAL, variable=self.lacunarity_var, command=self._update_noise_labels).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(wave_noise_frame, textvariable=self.lacunarity_display_var, width=5).grid(row=3, column=2, padx=5, pady=2)

        ttk.Label(wave_noise_frame, text="Amplitude:").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        self.amplitude_var = tk.DoubleVar(value=1.0)
        self.amplitude_display_var = tk.StringVar()
        ttk.Scale(wave_noise_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, variable=self.amplitude_var, command=self._update_noise_labels).grid(row=4, column=1, padx=5, pady=2)
        ttk.Label(wave_noise_frame, textvariable=self.amplitude_display_var, width=5).grid(row=4, column=2, padx=5, pady=2)

        # --- Frame de Detalhes das Ondas (Segunda Camada de Ruído) ---
        detail_noise_frame = ttk.LabelFrame(self.scrollable_frame, text="Detalhes das Ondas (Marolas)", padding="10")
        detail_noise_frame.pack(fill=tk.X, pady=5, anchor='n')

        ttk.Label(detail_noise_frame, text="Escala Detalhe:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.detail_scale_var = tk.DoubleVar(value=80.0) # Escala maior = detalhes menores/mais frequentes
        self.detail_scale_display_var = tk.StringVar()
        ttk.Scale(detail_noise_frame, from_=10.0, to=200.0, orient=tk.HORIZONTAL, variable=self.detail_scale_var, length=180, command=self._update_detail_noise_labels).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(detail_noise_frame, textvariable=self.detail_scale_display_var, width=5).grid(row=0, column=2, padx=5, pady=2)

        ttk.Label(detail_noise_frame, text="Oitavas Detalhe:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.detail_octaves_var = tk.IntVar(value=4)
        ttk.Scale(detail_noise_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.detail_octaves_var, command=lambda v: self.detail_octaves_var.set(int(float(v)))).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(detail_noise_frame, textvariable=self.detail_octaves_var, width=5).grid(row=1, column=2, padx=5, pady=2)

        ttk.Label(detail_noise_frame, text="Influência Detalhe:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.detail_influence_var = tk.DoubleVar(value=0.2) # Quanto o detalhe contribui
        self.detail_influence_display_var = tk.StringVar()
        ttk.Scale(detail_noise_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.detail_influence_var, command=self._update_detail_noise_labels).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(detail_noise_frame, textvariable=self.detail_influence_display_var, width=5).grid(row=2, column=2, padx=5, pady=2)

        # --- Frame de Aparência da Água ---
        appearance_frame = ttk.LabelFrame(self.scrollable_frame, text="Aparência da Água", padding="10")
        appearance_frame.pack(fill=tk.X, pady=5, anchor='n')
        # (Adicionaremos cor, espuma, etc. aqui)
        ttk.Label(appearance_frame, text="Cor Base:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.base_color_var = tk.StringVar(value="#204080") # Azul padrão
        self.base_color_button = tk.Button(appearance_frame, text=" ", bg=self.base_color_var.get(), width=4, relief=tk.GROOVE, command=self._choose_base_color)
        self.base_color_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(appearance_frame, textvariable=self.base_color_var).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        ttk.Label(appearance_frame, text="Cor Profunda:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.depth_color_var = tk.StringVar(value="#051030") # Azul mais escuro
        self.depth_color_button = tk.Button(appearance_frame, text=" ", bg=self.depth_color_var.get(), width=4, relief=tk.GROOVE, command=self._choose_depth_color)
        self.depth_color_button.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(appearance_frame, textvariable=self.depth_color_var).grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

        ttk.Label(appearance_frame, text="Nível Espuma:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.foam_level_var = tk.DoubleVar(value=0.6)
        self.foam_level_display_var = tk.StringVar()
        ttk.Scale(appearance_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.foam_level_var, command=self._update_appearance_labels).grid(row=2, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW)
        ttk.Label(appearance_frame, textvariable=self.foam_level_display_var, width=5).grid(row=2, column=3, padx=5, pady=2)

        ttk.Label(appearance_frame, text="Intens. Espuma:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.foam_intensity_var = tk.DoubleVar(value=0.8)
        self.foam_intensity_display_var = tk.StringVar()
        ttk.Scale(appearance_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.foam_intensity_var, command=self._update_appearance_labels).grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW)
        ttk.Label(appearance_frame, textvariable=self.foam_intensity_display_var, width=5).grid(row=3, column=3, padx=5, pady=2)

        appearance_frame.columnconfigure(1, weight=1) # Para scale expandir

        # --- Frame de Saída ---
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Mapas de Saída", padding="10")
        output_frame.pack(fill=tk.X, pady=5, anchor='n')
        self.gen_color_var = tk.BooleanVar(value=True)
        self.gen_normal_var = tk.BooleanVar(value=True)
        self.gen_roughness_var = tk.BooleanVar(value=True)
        self.gen_displacement_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="Cor (Albedo)", variable=self.gen_color_var).pack(anchor=tk.W)
        ttk.Checkbutton(output_frame, text="Normal Map", variable=self.gen_normal_var).pack(anchor=tk.W)
        ttk.Checkbutton(output_frame, text="Roughness Map", variable=self.gen_roughness_var).pack(anchor=tk.W)
        ttk.Checkbutton(output_frame, text="Displacement Map", variable=self.gen_displacement_var).pack(anchor=tk.W)


        # --- Frame de Controles (Gerar/Salvar/Ajuda) ---
        controls_frame = ttk.Frame(self.scrollable_frame, padding="10")
        controls_frame.pack(fill=tk.X, pady=10, anchor='n')
        self.generate_button = ttk.Button(controls_frame, text="Gerar Texturas", command=self.generate_textures)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(controls_frame, text="Salvar Mapas", command=self.save_maps, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.help_button = ttk.Button(controls_frame, text="Ajuda", command=self._show_help_window)
        self.help_button.pack(side=tk.RIGHT, padx=5)

        # Barra de progresso (escondida inicialmente)
        self.progress_bar = ttk.Progressbar(self.scrollable_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        # Não empacotar ainda, será feito durante a geração

        # --- Área de Visualização (Painel Direito) ---
        # Vamos usar um grid para mostrar múltiplos previews
        self.preview_frame = ttk.Frame(right_panel)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

        cols = 2
        rows = 2
        self.preview_labels = {}
        map_types = ["Color", "Normal", "Roughness", "Displacement"]
        for i, map_type in enumerate(map_types):
            r, c = divmod(i, cols)
            frame = ttk.Frame(self.preview_frame, borderwidth=1, relief="solid")
            frame.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            self.preview_frame.grid_rowconfigure(r, weight=1)
            self.preview_frame.grid_columnconfigure(c, weight=1)

            title_label = ttk.Label(frame, text=map_type, anchor=tk.CENTER)
            title_label.pack(pady=(2,0))
            img_label = ttk.Label(frame, text="Preview", anchor=tk.CENTER, background="gray") # Fundo cinza para destaque
            img_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
            self.preview_labels[map_type] = img_label

        # Armazenamento para imagens geradas (arrays numpy e PIL Images)
        self.generated_maps_numpy = {}
        self.generated_maps_pil = {}
        self.generated_maps_photo = {} # Para PhotoImage do Tkinter

        # Inicializar estados e labels
        self._update_noise_labels()
        self._update_appearance_labels()
        self._update_detail_noise_labels()
        self._check_gpu_availability()

    # --- Funções de Callback e UI ---
    def _update_noise_labels(self, event=None):
        self.scale_display_var.set(f"{self.scale_var.get():.1f}")
        self.persistence_display_var.set(f"{self.persistence_var.get():.2f}")
        self.lacunarity_display_var.set(f"{self.lacunarity_var.get():.1f}")
        self.amplitude_display_var.set(f"{self.amplitude_var.get():.2f}")

    def _update_appearance_labels(self, event=None):
        self.foam_level_display_var.set(f"{self.foam_level_var.get():.2f}")
        self.foam_intensity_display_var.set(f"{self.foam_intensity_var.get():.2f}")

    def _update_detail_noise_labels(self, event=None):
        """Atualiza os labels dos parâmetros de ruído de detalhe."""
        self.detail_scale_display_var.set(f"{self.detail_scale_var.get():.1f}")
        self.detail_influence_display_var.set(f"{self.detail_influence_var.get():.2f}")

    def randomize_seed(self):
        self.seed_var.set(random.randint(0, 99999))

    def _choose_color(self, target_var, button):
        """Abre o color chooser e atualiza a variável e o botão."""
        initial_color = target_var.get()
        color_code = colorchooser.askcolor(title="Escolher Cor", initialcolor=initial_color)
        if color_code and color_code[1]: # Verifica se o usuário não cancelou e recebeu uma cor válida
            target_var.set(color_code[1])
            button.config(bg=color_code[1])

    def _choose_base_color(self):
        self._choose_color(self.base_color_var, self.base_color_button)

    def _choose_depth_color(self):
        self._choose_color(self.depth_color_var, self.depth_color_button)

    def _check_gpu_availability(self):
        """Verifica GPU e atualiza UI."""
        if not NUMBA_AVAILABLE:
            self.gpu_status_label.config(text="Numba não instalado.", foreground="red")
            for child in self.proc_frame.winfo_children():
                if isinstance(child, ttk.Radiobutton) and child['value'] == "GPU":
                    child.config(state=tk.DISABLED)
            self.processing_device.set("CPU")
            return False

        try:
            if cuda.is_available():
                devices = cuda.list_devices()
                if devices:
                    gpu_name = devices[0].name.decode('UTF-8')
                    self.gpu_status_label.config(text=f"GPU: {gpu_name}", foreground="green")
                    for child in self.proc_frame.winfo_children():
                        if isinstance(child, ttk.Radiobutton) and child['value'] == "GPU":
                            child.config(state=tk.NORMAL)
                    return True
                else:
                    self.gpu_status_label.config(text="Nenhuma GPU CUDA encontrada.", foreground="orange")
            else:
                self.gpu_status_label.config(text="CUDA não disponível.", foreground="red")
        except Exception as e:
            self.gpu_status_label.config(text=f"Erro ao verificar GPU.", foreground="red")
            print(f"Erro detalhado ao verificar GPU: {traceback.format_exc()}")

        # Desabilita GPU se não disponível/erro
        for child in self.proc_frame.winfo_children():
             if isinstance(child, ttk.Radiobutton) and child['value'] == "GPU":
                 child.config(state=tk.DISABLED)
        if self.processing_device.get() == "GPU":
             self.processing_device.set("CPU")
        return False

    # --- Funções de Geração ---
    def generate_textures(self):
        """Ponto de entrada para a geração, decide entre CPU e GPU."""
        device = self.processing_device.get()
        if device == "GPU":
            if not NUMBA_AVAILABLE:
                messagebox.showerror("Erro", "Numba não está instalado. Impossível usar GPU.")
                return
            if not self._check_gpu_availability(): # Re-verifica
                messagebox.showerror("Erro", "GPU não disponível. Usando CPU.")
                self.processing_device.set("CPU")
                self._generate_textures_cpu()
            else:
                self._generate_textures_gpu()
        else:
            self._generate_textures_cpu()

    def _generate_textures_cpu(self):
        """Gera as texturas na CPU."""
        start_time = time.time()
        self.generate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, pady=(10, 5), anchor='n')
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        self.root.update_idletasks()

        try:
            # Obter parâmetros da UI
            width = self.width_var.get()
            height = self.height_var.get()
            if width <= 0 or height <= 0:
                raise ValueError("Largura e Altura devem ser positivas.")

            seed = self.seed_var.get()
            scale = self.scale_var.get()
            octaves = self.octaves_var.get()
            persistence = self.persistence_var.get()
            lacunarity = self.lacunarity_var.get()
            amplitude_mult = self.amplitude_var.get() # Amplitude do deslocamento

            # Parâmetros do Ruído de Detalhe
            detail_scale = self.detail_scale_var.get()
            detail_octaves = self.detail_octaves_var.get()
            detail_influence = self.detail_influence_var.get()
            # Usaremos a mesma persistencia/lacunaridade/seed+1 para o detalhe
            detail_persistence = persistence
            detail_lacunarity = lacunarity
            detail_seed = seed + 1 # Semente diferente para o detalhe

            base_color_hex = self.base_color_var.get()
            depth_color_hex = self.depth_color_var.get()
            foam_level = self.foam_level_var.get()
            foam_intensity = self.foam_intensity_var.get()

            gen_color = self.gen_color_var.get()
            gen_normal = self.gen_normal_var.get()
            gen_roughness = self.gen_roughness_var.get()
            gen_displacement = self.gen_displacement_var.get()

            print(f"Gerando texturas (CPU): {width}x{height}, seed={seed}")

            # 1. Gerar Ruído Base (Ondas Maiores)
            print(" - Gerando ruído base (ondas maiores)...")
            rng_main = np.random.RandomState(seed)
            perm_main = rng_main.permutation(256)
            perm_table_main = np.concatenate((perm_main, perm_main)).astype(np.int32)
            noise_map_base = noise_loop_cpu(width, height, scale, octaves, persistence, lacunarity, seed, perm_table_main)

            # 1.1 Gerar Ruído de Detalhe (Marolas)
            print(" - Gerando ruído de detalhe (marolas)...")
            rng_detail = np.random.RandomState(detail_seed)
            perm_detail = rng_detail.permutation(256)
            perm_table_detail = np.concatenate((perm_detail, perm_detail)).astype(np.int32)
            noise_map_detail = noise_loop_cpu(width, height, detail_scale, detail_octaves, detail_persistence, detail_lacunarity, detail_seed, perm_table_detail)

            # 1.2 Combinar os Ruídos
            # Normaliza ambos para [-1, 1] antes de combinar com peso
            min_base, max_base = np.min(noise_map_base), np.max(noise_map_base)
            min_detail, max_detail = np.min(noise_map_detail), np.max(noise_map_detail)

            if max_base > min_base:
                noise_base_norm = (noise_map_base - min_base) / (max_base - min_base) * 2.0 - 1.0
            else:
                noise_base_norm = np.zeros_like(noise_map_base)

            if max_detail > min_detail:
                noise_detail_norm = (noise_map_detail - min_detail) / (max_detail - min_detail) * 2.0 - 1.0
            else:
                noise_detail_norm = np.zeros_like(noise_map_detail)

            # Combina usando a influência como peso para o detalhe
            combined_noise = noise_base_norm * (1.0 - detail_influence) + noise_detail_norm * detail_influence

            # Normalizar ruído COMBINADO para [0, 1] para usar em cor/espuma/rugosidade
            min_combined, max_combined = np.min(combined_noise), np.max(combined_noise)
            if max_combined > min_combined:
                noise_normalized_01 = (combined_noise - min_combined) / (max_combined - min_combined)
            else:
                noise_normalized_01 = np.full((height, width), 0.5, dtype=np.float64)

            # Calcular displacement map a partir do ruído combinado [-1, 1] e amplitude geral
            displacement_map = combined_noise * amplitude_mult

            self.generated_maps_numpy["Displacement"] = displacement_map if gen_displacement else None

            # 2. Calcular Mapa de Normais (a partir do displacement combinado)
            if gen_normal:
                print(" - Calculando mapa de normais...")
                normal_map = self._calculate_normal_map_cpu(displacement_map)
                self.generated_maps_numpy["Normal"] = normal_map
            else:
                 self.generated_maps_numpy["Normal"] = None

            # 3. Calcular Mapa de Rugosidade
            if gen_roughness:
                print(" - Calculando mapa de rugosidade...")
                # Áreas de espuma (picos de ondas) são mais ásperas
                # Vamos usar o noise_normalized_01 > foam_level como base
                foam_mask = (noise_normalized_01 > foam_level).astype(np.float64)
                # Rugosidade base baixa (água lisa) + rugosidade alta na espuma
                base_roughness = 0.05
                foam_roughness = 0.8
                roughness_map = lerp_cpu(foam_mask, base_roughness, foam_roughness)
                self.generated_maps_numpy["Roughness"] = roughness_map # Valores [0, 1]
            else:
                 self.generated_maps_numpy["Roughness"] = None

            # 4. Calcular Mapa de Cor (Albedo)
            if gen_color:
                print(" - Calculando mapa de cores...")
                color_map = self._calculate_color_map_cpu(noise_normalized_01, base_color_hex, depth_color_hex, foam_level, foam_intensity)
                self.generated_maps_numpy["Color"] = color_map # Valores [0, 255] uint8 RGB
            else:
                 self.generated_maps_numpy["Color"] = None


            # Converter numpy arrays para PIL Images para exibição/salvamento
            self._convert_numpy_to_pil()

            # Exibir previews
            self._display_previews()

            self.save_button.config(state=tk.NORMAL)
            end_time = time.time()
            print(f"Texturas (CPU) geradas em {end_time - start_time:.2f} segundos.")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erro na Geração (CPU)", f"Ocorreu um erro: {e}")
            self.save_button.config(state=tk.DISABLED)
        finally:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.generate_button.config(state=tk.NORMAL)
            self.root.update_idletasks()

    def _generate_textures_gpu(self):
        """Gera as texturas na GPU (CUDA)."""
        start_time = time.time()
        self.generate_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, pady=(10, 5), anchor='n')
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        self.root.update_idletasks()

        # Limpa mapas antigos
        self.generated_maps_numpy = {}
        self.generated_maps_pil = {}
        self.generated_maps_photo = {}

        try:
            # Obter parâmetros da UI
            width = self.width_var.get()
            height = self.height_var.get()
            if width <= 0 or height <= 0:
                raise ValueError("Largura e Altura devem ser positivas.")

            seed = self.seed_var.get()
            scale = self.scale_var.get()
            octaves = self.octaves_var.get()
            persistence = self.persistence_var.get()
            lacunarity = self.lacunarity_var.get()
            amplitude_mult = self.amplitude_var.get()

            # Parâmetros do Ruído de Detalhe
            detail_scale = self.detail_scale_var.get()
            detail_octaves = self.detail_octaves_var.get()
            detail_influence = self.detail_influence_var.get()
            detail_persistence = persistence
            detail_lacunarity = lacunarity
            detail_seed = seed + 1

            base_color_hex = self.base_color_var.get()
            depth_color_hex = self.depth_color_var.get()
            base_r, base_g, base_b = self._hex_to_rgb(base_color_hex)
            depth_r, depth_g, depth_b = self._hex_to_rgb(depth_color_hex)

            foam_level = self.foam_level_var.get()
            foam_intensity = self.foam_intensity_var.get()
            base_roughness = 0.05 # Hardcoded por enquanto
            foam_roughness = 0.8  # Hardcoded por enquanto

            gen_color = self.gen_color_var.get()
            gen_normal = self.gen_normal_var.get()
            gen_roughness = self.gen_roughness_var.get()
            gen_displacement = self.gen_displacement_var.get()

            print(f"Gerando texturas (GPU Completo): {width}x{height}, seed={seed}")

            # Configuração CUDA
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(width / threadsperblock[0])
            blockspergrid_y = math.ceil(height / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            # 1. Gerar Ruído Base (Ondas Maiores) na GPU
            print(" - Gerando ruído base (GPU)...")
            rng_main = np.random.RandomState(seed)
            perm_main = rng_main.permutation(256)
            perm_table_main = np.concatenate((perm_main, perm_main)).astype(np.int32)
            perm_table_main_gpu = cuda.to_device(perm_table_main)
            noise_map_base_gpu = cuda.device_array((height, width), dtype=np.float64)
            noise_kernel_gpu[blockspergrid, threadsperblock](
                noise_map_base_gpu, width, height, scale, octaves, persistence, lacunarity, seed, perm_table_main_gpu
            )
            # cuda.synchronize() # Sincronizar após todos os lançamentos independentes

            # 1.1 Gerar Ruído de Detalhe (Marolas) na GPU
            print(" - Gerando ruído de detalhe (GPU)...")
            rng_detail = np.random.RandomState(detail_seed)
            perm_detail = rng_detail.permutation(256)
            perm_table_detail = np.concatenate((perm_detail, perm_detail)).astype(np.int32)
            perm_table_detail_gpu = cuda.to_device(perm_table_detail)
            noise_map_detail_gpu = cuda.device_array((height, width), dtype=np.float64)
            noise_kernel_gpu[blockspergrid, threadsperblock](
                noise_map_detail_gpu, width, height, detail_scale, detail_octaves, detail_persistence, detail_lacunarity, detail_seed, perm_table_detail_gpu
            )

            cuda.synchronize() # Espera ambos os kernels de ruído terminarem

            # --- Normalização e Combinação --- 
            # Para combinar corretamente com a influência, normalizamos ambos para [-1, 1]
            # Isso ainda requer min/max (copiando para CPU por enquanto)
            print(" - Normalizando ruídos base e detalhe (GPU)... ")
            noise_base_cpu = noise_map_base_gpu.copy_to_host()
            noise_detail_cpu = noise_map_detail_gpu.copy_to_host()
            min_base, max_base = np.min(noise_base_cpu), np.max(noise_base_cpu)
            min_detail, max_detail = np.min(noise_detail_cpu), np.max(noise_detail_cpu)
            range_base = max_base - min_base
            range_detail = max_detail - min_detail
            del noise_base_cpu, noise_detail_cpu # Libera memória CPU

            # Aloca memória na GPU para os ruídos normalizados [-1, 1]
            noise_base_norm_gpu = cuda.device_array((height, width), dtype=np.float64)
            noise_detail_norm_gpu = cuda.device_array((height, width), dtype=np.float64)

            # Kernel de normalização adaptado para [-1, 1] (poderia ser um kernel separado)
            # (input - min) / range -> [0, 1]
            # ((input - min) / range) * 2.0 - 1.0 -> [-1, 1]
            # (input - min) * (2.0 / range) - 1.0
            if range_base > 1e-6:
                 normalize_kernel_gpu[blockspergrid, threadsperblock](
                     noise_map_base_gpu, noise_base_norm_gpu, width, height, min_base, range_base / 2.0 # Divide range por 2
                 )
                 # Subtrai 1 (pode ser outro kernel ou feito no combine)
            else: # Se constante, normaliza para 0
                 noise_base_norm_gpu.copy_to_device(np.zeros((height, width), dtype=np.float64))
            
            if range_detail > 1e-6:
                 normalize_kernel_gpu[blockspergrid, threadsperblock](
                     noise_map_detail_gpu, noise_detail_norm_gpu, width, height, min_detail, range_detail / 2.0
                 )
            else:
                 noise_detail_norm_gpu.copy_to_device(np.zeros((height, width), dtype=np.float64))
            
            cuda.synchronize()
            # TODO: Subtrair 1 dos arrays normalizados ou ajustar combine_kernel
            # Por ora, a combinação funcionará, mas o range estará deslocado.

            # 1.2 Combinar os Ruídos Normalizados [-1, 1] na GPU
            print(" - Combinando ruídos (GPU)...")
            combined_noise_gpu = cuda.device_array((height, width), dtype=np.float64)
            combine_noise_kernel_gpu[blockspergrid, threadsperblock](
                noise_base_norm_gpu, noise_detail_norm_gpu, combined_noise_gpu, width, height, detail_influence
            )
            cuda.synchronize()

            # 2. Normalizar ruído COMBINADO para [0, 1] na GPU (para cor/rugosidade)
            print(" - Normalizando ruído combinado [0,1] (GPU)...")
            # Requer min/max do combinado
            combined_noise_cpu = combined_noise_gpu.copy_to_host()
            min_combined, max_combined = np.min(combined_noise_cpu), np.max(combined_noise_cpu)
            range_combined = max_combined - min_combined
            del combined_noise_cpu

            noise_final_norm_01_gpu = cuda.device_array((height, width), dtype=np.float64)
            if range_combined > 1e-6:
                 normalize_kernel_gpu[blockspergrid, threadsperblock](
                     combined_noise_gpu, noise_final_norm_01_gpu, width, height, min_combined, range_combined
                 )
            else:
                 noise_final_norm_01_gpu.copy_to_device(np.full((height, width), 0.5, dtype=np.float64))
            cuda.synchronize()

            # 3. Calcular Mapa de Deslocamento (CPU, baseado no combinado [-1,1] e amplitude)
            displacement_map_host = None
            if gen_displacement:
                print(" - Calculando displacement map (CPU)...")
                # Usa o combined_noise_gpu que já está em [-A, A] (aproximadamente, devido à normalização [-1,1])
                # Para aplicar amplitude_mult corretamente, copiamos o combinado e multiplicamos
                combined_noise_host = combined_noise_gpu.copy_to_host()
                displacement_map_host = combined_noise_host * amplitude_mult
                self.generated_maps_numpy["Displacement"] = displacement_map_host
                del combined_noise_host
            else:
                self.generated_maps_numpy["Displacement"] = None

            # 4. Calcular Mapa de Normais (GPU, a partir do displacement)
            normal_map_gpu = None
            if gen_normal:
                print(" - Calculando mapa de normais (GPU)...")
                normal_map_gpu = cuda.device_array((height, width, 3), dtype=np.uint8)
                
                # Precisamos do displacement map na GPU.
                # Se foi gerado, copiamos da CPU para GPU. Se não, usamos o combinado?
                # Opção 1: Copiar displacement_map_host para GPU se existir.
                if displacement_map_host is not None:
                    displacement_map_gpu = cuda.to_device(displacement_map_host)
                else: 
                    # Se displacement não foi gerado, calcula normais do ruído combinado sem amplitude?
                    # Ou aplica amplitude ao combined_noise_gpu?
                    # Vamos usar combined_noise_gpu diretamente como altura relativa
                    displacement_map_gpu = combined_noise_gpu # Usando ruído combinado como altura
                
                normal_map_kernel_gpu[blockspergrid, threadsperblock](
                    displacement_map_gpu, normal_map_gpu, width, height
                )
                cuda.synchronize()
                self.generated_maps_numpy["Normal"] = normal_map_gpu.copy_to_host()
                # Limpa displacement_map_gpu se foi criado aqui
                if displacement_map_host is not None: 
                    del displacement_map_gpu 
            else:
                self.generated_maps_numpy["Normal"] = None

            # 5. Calcular Mapa de Rugosidade (GPU, a partir do combinado normalizado [0,1])
            roughness_map_gpu = None
            if gen_roughness:
                print(" - Calculando mapa de rugosidade (GPU)...")
                roughness_map_gpu = cuda.device_array((height, width), dtype=np.uint8)
                roughness_map_kernel_gpu[blockspergrid, threadsperblock](
                    noise_final_norm_01_gpu, roughness_map_gpu, width, height, # Usa o combinado norm [0,1]
                    foam_level, base_roughness, foam_roughness
                )
                cuda.synchronize()
                self.generated_maps_numpy["Roughness"] = roughness_map_gpu.copy_to_host()
            else:
                self.generated_maps_numpy["Roughness"] = None

            # 6. Calcular Mapa de Cor (GPU, a partir do combinado normalizado [0,1])
            color_map_gpu = None
            if gen_color:
                print(" - Calculando mapa de cores (GPU)...")
                color_map_gpu = cuda.device_array((height, width, 3), dtype=np.uint8)
                color_map_kernel_gpu[blockspergrid, threadsperblock](
                    noise_final_norm_01_gpu, color_map_gpu, width, height, # Usa o combinado norm [0,1]
                    base_r, base_g, base_b, depth_r, depth_g, depth_b,
                    foam_level, foam_intensity
                )
                cuda.synchronize()
                self.generated_maps_numpy["Color"] = color_map_gpu.copy_to_host()
            else:
                self.generated_maps_numpy["Color"] = None

            # Limpa memória da GPU 
            del perm_table_main_gpu, perm_table_detail_gpu
            del noise_map_base_gpu, noise_map_detail_gpu
            del noise_base_norm_gpu, noise_detail_norm_gpu
            del combined_noise_gpu
            del noise_final_norm_01_gpu
            if normal_map_gpu: del normal_map_gpu
            if roughness_map_gpu: del roughness_map_gpu
            if color_map_gpu: del color_map_gpu
            # Verifica se displacement_map_gpu foi criada e precisa ser deletada
            if gen_normal and displacement_map_host is not None and 'displacement_map_gpu' in locals():
                 del displacement_map_gpu

            # Converter e exibir 
            self._convert_numpy_to_pil()
            self._display_previews()

            self.save_button.config(state=tk.NORMAL)
            end_time = time.time()
            print(f"Texturas (GPU Completo) geradas em {end_time - start_time:.2f} segundos.")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Erro na Geração (GPU)", f"Ocorreu um erro: {e}")
            self.save_button.config(state=tk.DISABLED)
        finally:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.generate_button.config(state=tk.NORMAL)
            self.root.update_idletasks()

    # --- Funções de Cálculo de Mapas (CPU) ---

    def _hex_to_rgb(self, hex_color):
        """Converte #RRGGBB para tuple (R, G, B) int 0-255."""
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _calculate_color_map_cpu(self, noise_01, base_hex, depth_hex, foam_thresh, foam_intensity):
        """Calcula o mapa de cores com base no ruído, cores e espuma."""
        height, width = noise_01.shape
        color_map = np.zeros((height, width, 3), dtype=np.uint8)

        base_rgb = np.array(self._hex_to_rgb(base_hex), dtype=np.float64)
        depth_rgb = np.array(self._hex_to_rgb(depth_hex), dtype=np.float64)
        white = np.array([255.0, 255.0, 255.0], dtype=np.float64)

        # Interpola entre cor profunda (noise=0) e cor base (noise=1)
        water_color = lerp_cpu(noise_01[..., np.newaxis], depth_rgb, base_rgb)

        # Adiciona espuma onde o ruído excede o limiar
        foam_mask = (noise_01 > foam_thresh).astype(np.float64)
        # Interpola entre a cor da água e branco baseado na intensidade da espuma
        # A interpolação é feita apenas onde foam_mask > 0
        final_color = lerp_cpu(foam_mask[..., np.newaxis] * foam_intensity, water_color, white)

        color_map = np.clip(final_color, 0, 255).astype(np.uint8)
        return color_map

    def _calculate_normal_map_cpu(self, displacement_map):
        """Calcula o mapa de normais a partir do mapa de deslocamento (altura)."""
        height, width = displacement_map.shape
        # Calcula gradientes (diferenças entre pixels vizinhos)
        # Usamos np.gradient que lida com as bordas
        dz_dy, dz_dx = np.gradient(displacement_map)

        # O vetor normal é perpendicular ao gradiente (-dz/dx, -dz/dy, 1)
        # Componentes precisam ser normalizados para o espaço de cores [0, 255]
        # X -> Vermelho, Y -> Verde, Z -> Azul
        # Mapeamento: Componente de [-1, 1] para [0, 255] -> (componente + 1) * 0.5 * 255
        normal_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Cria vetores não normalizados
        normals = np.dstack((-dz_dx, -dz_dy, np.ones((height, width))))

        # Normaliza os vetores
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        # Evita divisão por zero
        valid_norms = norms > 1e-6
        # normals[valid_norms] /= norms[valid_norms] # Linha original com erro
        # Usa np.divide com 'where' para normalizar apenas vetores válidos
        np.divide(normals, norms, out=normals, where=valid_norms)

        # Mapeia componentes para [0, 255]
        # Componente X (-dz_dx normalizado) -> Canal R
        normal_map[..., 0] = ((normals[..., 0] + 1.0) * 0.5 * 255).astype(np.uint8)
        # Componente Y (-dz_dy normalizado) -> Canal G
        # No OpenGL/DirectX, a coordenada Y do espaço de textura pode apontar para baixo,
        # então frequentemente o componente Y do normal map é invertido (1 - y).
        # Vamos seguir essa convenção.
        normal_map[..., 1] = (((-normals[..., 1]) + 1.0) * 0.5 * 255).astype(np.uint8)
        # Componente Z (sempre positivo após normalização) -> Canal B
        normal_map[..., 2] = ((normals[..., 2]) * 255).astype(np.uint8) # Mapeia [0, 1] para [0, 255]

        return normal_map

    # --- Funções de Conversão e Exibição ---
    def _convert_numpy_to_pil(self):
        """Converte os numpy arrays gerados para PIL Images."""
        self.generated_maps_pil = {}
        for map_type, np_array in self.generated_maps_numpy.items():
            if np_array is None:
                self.generated_maps_pil[map_type] = None
                continue

            try:
                if map_type == "Color": # RGB uint8
                    img = Image.fromarray(np_array, 'RGB')
                elif map_type == "Normal": # RGB uint8
                    img = Image.fromarray(np_array, 'RGB')
                elif map_type == "Roughness": # Grayscale float [0, 1] -> uint8 [0, 255]
                    img_array = (np.clip(np_array, 0.0, 1.0) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, 'L')
                elif map_type == "Displacement": # Grayscale float [-A, A] -> uint8 [0, 255]
                    min_val, max_val = np.min(np_array), np.max(np_array)
                    if max_val > min_val:
                        norm_array = (np_array - min_val) / (max_val - min_val)
                    else:
                        norm_array = np.zeros_like(np_array)
                    img_array = (norm_array * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, 'L')
                else:
                    print(f"Aviso: Tipo de mapa desconhecido para conversão PIL: {map_type}")
                    img = None

                self.generated_maps_pil[map_type] = img
            except Exception as e:
                print(f"Erro ao converter mapa '{map_type}' para PIL: {e}")
                self.generated_maps_pil[map_type] = None


    def _display_previews(self):
        """Exibe as imagens PIL geradas nos labels de preview."""
        self.generated_maps_photo = {} # Limpa refs antigas

        for map_type, pil_image in self.generated_maps_pil.items():
            label = self.preview_labels.get(map_type)
            if label is None: continue

            if pil_image is None:
                # Limpa o preview se o mapa não foi gerado
                label.config(image='', text=f"{map_type} (Não gerado)", background="lightgrey")
                label.image = None
                continue

            # Redimensiona para caber no label
            label.update_idletasks() # Garante que temos as dimensões corretas
            lbl_width = label.winfo_width()
            lbl_height = label.winfo_height()
            if lbl_width < 10 or lbl_height < 10: # Fallback se o label ainda não foi desenhado
                lbl_width, lbl_height = 150, 150

            img_width, img_height = pil_image.size
            aspect_ratio = img_width / img_height

            # Calcula dimensões mantendo aspect ratio
            new_width = lbl_width
            new_height = int(new_width / aspect_ratio)
            if new_height > lbl_height:
                new_height = lbl_height
                new_width = int(new_height * aspect_ratio)

            # Evita tamanho zero ou negativo
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            try:
                img_resized = pil_image.resize((new_width, new_height), Image.Resampling.NEAREST)
                photo_image = ImageTk.PhotoImage(img_resized)

                label.config(image=photo_image, text="") # Remove texto placeholder
                label.image = photo_image # Mantem referência
                self.generated_maps_photo[map_type] = photo_image # Guarda referência globalmente também
            except Exception as e:
                 print(f"Erro ao redimensionar ou exibir preview para {map_type}: {e}")
                 label.config(image='', text=f"{map_type} (Erro Display)", background="red")
                 label.image = None


    # --- Funções de Salvamento e Ajuda ---
    def save_maps(self):
        """Salva os mapas gerados."""
        if not self.generated_maps_pil or all(v is None for v in self.generated_maps_pil.values()):
            messagebox.showwarning("Aviso", "Nenhum mapa foi gerado para salvar.")
            return

        # Pede um nome base ao usuário
        seed = self.seed_var.get()
        width = self.width_var.get()
        height = self.height_var.get()
        default_filename = f"water_{width}x{height}_seed{seed}"

        base_filepath = filedialog.asksaveasfilename(
            title="Salvar Mapas Como (Nome Base)",
            initialfile=default_filename,
            defaultextension=".png", # Será adicionado se não tiver extensão
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if not base_filepath:
            return # Usuário cancelou

        # Extrai diretório e nome base sem extensão
        import os
        save_dir = os.path.dirname(base_filepath)
        base_name = os.path.splitext(os.path.basename(base_filepath))[0]

        saved_count = 0
        errors = []

        for map_type, pil_image in self.generated_maps_pil.items():
            if pil_image is None:
                continue # Não salva mapas não gerados

            # Monta o nome final do arquivo
            output_filename = f"{base_name}_{map_type.lower()}.png"
            output_path = os.path.join(save_dir, output_filename)

            try:
                pil_image.save(output_path, "PNG")
                print(f"Mapa '{map_type}' salvo em: {output_path}")
                saved_count += 1
            except Exception as e:
                error_msg = f"Erro ao salvar '{map_type}' em {output_path}: {e}"
                print(error_msg)
                errors.append(error_msg)

        if saved_count > 0 and not errors:
            messagebox.showinfo("Sucesso", f"{saved_count} mapa(s) salvo(s) com sucesso começando com '{base_name}'.")
        elif errors:
            # Constrói a mensagem de erro detalhada primeiro
            error_details = "\n- " + "\n- ".join(errors)
            error_message = f"{saved_count} mapa(s) salvo(s), mas ocorreram erros:\n{error_details}"
            messagebox.showerror("Erro ao Salvar", error_message)


    def _show_help_window(self):
        """Exibe a janela de ajuda (Placeholder)."""
        # TODO: Adaptar o texto da ajuda do gerador de heightmap para este script.
        help_text = """Instruções de uso:

- Ajuste os parâmetros de ruído e aparência.
- Escolha CPU ou GPU para processamento.
- Clique em 'Gerar Texturas'.
- Os previews dos mapas selecionados aparecerão.
- Use 'Salvar Mapas' para salvar os arquivos PNG."""
        messagebox.showinfo("Ajuda", help_text)

# --- Ponto de Entrada ---
if __name__ == "__main__":
    if not NUMBA_AVAILABLE:
        print("AVISO: Numba não encontrado. Funcionalidade GPU e Perlin CPU estarão desabilitadas.")

    root = tk.Tk()
    app = WaterTextureGeneratorApp(root)
    # Chama a função de display inicial para ajustar o tamanho dos labels
    root.after(100, app._display_previews)
    root.mainloop() 