import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
from opensimplex import OpenSimplex
import random
import time # Adicionar time para medir performance
import traceback # Para traceback detalhado no check_gpu
import math # Pode ser útil para kernels

# Import Numba e CUDA
try:
    from numba import cuda, jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Definir stubs se numba não estiver disponível para evitar erros posteriores
    # nos decoradores se alguém tentar rodar sem numba
    class cuda:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def jit(func): return func # Retorna a função original
    def jit(options={}): 
        def decorator(func):
             return func
        return decorator
    njit = jit
    prange = range

# ======================================================
# Funções e Kernel CUDA para Geração de Ruído Perlin
# ======================================================

# --- Funções Auxiliares (Device Functions) ---
# Estas funções rodam na GPU e são chamadas pelo kernel.

@cuda.jit(device=True)
def fade(t):
    """Função de interpolação suave (6t^5 - 15t^4 + 10t^3)."""
    return t * t * t * (t * (t * 6 - 15) + 10)

@cuda.jit(device=True)
def lerp(t, a, b):
    """Interpolação linear."""
    return a + t * (b - a)

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
    # Encontra as coordenadas da célula do grid
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    
    # Encontra as coordenadas relativas dentro da célula
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    
    # Calcula as curvas de fade para interpolação suave
    u = fade(xf)
    v = fade(yf)
    
    # Hash das coordenadas dos 4 cantos da célula usando a tabela de permutação
    # A tabela perm_table deve ter tamanho 512 (perm[0..255] repetida)
    p = perm_table # Alias para encurtar
    aa = p[p[xi ] + yi ]
    ab = p[p[xi ] + yi + 1]
    ba = p[p[xi + 1] + yi ]
    bb = p[p[xi + 1] + yi + 1]
    
    # Calcula os produtos escalares e interpola
    g_aa = grad(aa, xf, yf)
    g_ba = grad(ba, xf - 1, yf)
    g_ab = grad(ab, xf, yf - 1)
    g_bb = grad(bb, xf - 1, yf - 1)
    
    # Interpolação
    x1 = lerp(u, g_aa, g_ba)
    x2 = lerp(u, g_ab, g_bb)
    return lerp(v, x1, x2)

# --- Kernel CUDA Principal ---
@cuda.jit
def perlin_octave_kernel(output_array, width, height, scale, octaves, persistence, lacunarity, seed, warp_enabled, warp_amplitude, warp_frequency, perm_table, warp_perm_table_x, warp_perm_table_y):
    """Kernel CUDA para gerar ruído Perlin com oitavas e domain warping."""
    # Calcula o índice global do thread (pixel)
    idx, idy = cuda.grid(2)

    if idx >= width or idy >= height:
        return # Thread fora dos limites da imagem

    # --- Coordenadas Base ---
    # Normaliza para [0, 1] inicialmente
    nx_base = float(idx) / width
    ny_base = float(idy) / height
    
    # --- Domain Warping (Opcional) ---
    # Usa tabelas de permutação separadas para warping
    nx_warped = nx_base
    ny_warped = ny_base
    if warp_enabled:
        amplitude_factor = 100.0 # Fator de escala para amplitude da distorção
        warp_x_noise = perlin_noise_2d(nx_base * warp_frequency, ny_base * warp_frequency, warp_perm_table_x)
        warp_y_noise = perlin_noise_2d(nx_base * warp_frequency, ny_base * warp_frequency, warp_perm_table_y)
        nx_warped += warp_x_noise * (warp_amplitude / amplitude_factor)
        ny_warped += warp_y_noise * (warp_amplitude / amplitude_factor)

    # --- Loop de Oitavas --- 
    # Usa as coordenadas (distorcidas ou não) e a escala principal
    total_noise = 0.0
    frequency = 1.0
    amplitude = 1.0
    # A escala principal controla o "zoom" inicial
    effective_scale = scale / 50.0 

    for k in range(octaves):
        noise_val = perlin_noise_2d(nx_warped * effective_scale * frequency, 
                                      ny_warped * effective_scale * frequency, 
                                      perm_table)
        total_noise += noise_val * amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Escreve o resultado no array de saída
    # O valor estará aproximadamente em [-1, 1] (ou um pouco mais dependendo da persistência)
    # A normalização para [0, 255] será feita na CPU depois
    output_array[idy, idx] = total_noise

# --- Kernel CUDA Principal V2 (agora sem ridged interno) ---
@cuda.jit
def perlin_octave_kernel_v2(output_array, width, height, world_scale, octaves, persistence, lacunarity, seed, 
                            warp_enabled, warp_amplitude, warp_frequency, 
                            island_mask_enabled, island_falloff, 
                            perm_table, warp_perm_table_x, warp_perm_table_y):
    idx, idy = cuda.grid(2)
    if idx >= width or idy >= height:
        return

    nx_base = float(idx) / width
    ny_base = float(idy) / height
    nx_warped = nx_base
    ny_warped = ny_base
    if warp_enabled:
        amplitude_factor = 100.0
        warp_x_noise = perlin_noise_2d(nx_base * warp_frequency, ny_base * warp_frequency, warp_perm_table_x)
        warp_y_noise = perlin_noise_2d(nx_base * warp_frequency, ny_base * warp_frequency, warp_perm_table_y)
        nx_warped += warp_x_noise * (warp_amplitude / amplitude_factor)
        ny_warped += warp_y_noise * (warp_amplitude / amplitude_factor)
        
    nx_base_scaled = nx_warped * world_scale 
    ny_base_scaled = ny_warped * world_scale

    total_noise = 0.0
    frequency = 1.0
    amplitude = 1.0
    
    for k in range(octaves):
        noise_val = perlin_noise_2d(nx_base_scaled * frequency,
                                      ny_base_scaled * frequency,
                                      perm_table)
        total_noise += noise_val * amplitude
        amplitude *= persistence
        frequency *= lacunarity
        
    # --- Aplica Máscara de Ilha (GPU) ---
    final_value = total_noise # Começa com ruído normal
    if island_mask_enabled:
        center_x = width / 2.0
        center_y = height / 2.0
        dist_x = (float(idx) - center_x) / center_x if center_x != 0 else 0.0
        dist_y = (float(idy) - center_y) / center_y if center_y != 0 else 0.0
        distance = math.sqrt(dist_x**2 + dist_y**2)
        distance = min(max(distance, 0.0), 1.0)
        falloff_adjusted = max(0.01, island_falloff)
        mask = 1.0 - math.pow(distance, 2.0 / falloff_adjusted)
        mask = min(max(mask, 0.0), 1.0)
        final_value *= mask 

    output_array[idy, idx] = final_value

# ======================================================
# Funções Numba @njit para Geração de Ruído Perlin (CPU)
# ======================================================
# Reutiliza as mesmas lógicas, mas compiladas para CPU com @njit

# --- Funções Auxiliares (Compiladas para CPU) ---
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

# --- Função Principal Perlin CPU (Paralelizada com Numba) ---
@njit(parallel=True)
def perlin_cpu_octave_loop(width, height, world_scale, octaves, persistence, lacunarity, seed, 
                           warp_enabled, warp_amplitude, warp_frequency, 
                           perm_table, warp_perm_table_x, warp_perm_table_y):
    """Loop principal para gerar ruído Perlin na CPU com Numba (paralelizado)."""
    output_array = np.zeros((height, width), dtype=np.float64)
    amplitude_factor = 100.0

    for i in prange(height):
        for j in range(width):
            nx_base = float(j) / width
            ny_base = float(i) / height
            nx_warped = nx_base
            ny_warped = ny_base
            if warp_enabled:
                warp_x_noise = perlin_noise_2d_cpu(nx_base * warp_frequency, ny_base * warp_frequency, warp_perm_table_x)
                warp_y_noise = perlin_noise_2d_cpu(nx_base * warp_frequency, ny_base * warp_frequency, warp_perm_table_y)
                nx_warped += warp_x_noise * (warp_amplitude / amplitude_factor)
                ny_warped += warp_y_noise * (warp_amplitude / amplitude_factor)

            nx_base_scaled = nx_warped * world_scale
            ny_base_scaled = ny_warped * world_scale

            total_noise = 0.0
            frequency = 1.0
            amplitude = 1.0
            
            for k in range(octaves):
                noise_val = perlin_noise_2d_cpu(nx_base_scaled * frequency,
                                                ny_base_scaled * frequency,
                                                perm_table)
                total_noise += noise_val * amplitude
                amplitude *= persistence
                frequency *= lacunarity
            output_array[i, j] = total_noise
            
    return output_array

# ======================================================

class HeightmapGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gerador de Heightmap")
        self.root.geometry("1280x720") 

        style = ttk.Style()
        style.theme_use('clam')

        # --- Frame Principal Esquerdo (agora contém Canvas + Scrollbar) ---
        left_frame_container = ttk.Frame(root)
        left_frame_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), pady=5)
        
        # Canvas para a área rolável
        canvas = tk.Canvas(left_frame_container, borderwidth=0)
        # Frame interno que conterá todos os controles
        self.scrollable_frame = ttk.Frame(canvas, padding="10") 

        # Scrollbar vertical
        scrollbar = ttk.Scrollbar(left_frame_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Empacota Scrollbar e Canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Adiciona o frame rolável ao Canvas
        canvas_frame_id = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # --- Função para atualizar a região de rolagem ---
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Ajusta a largura do frame rolável à largura do canvas
            canvas.itemconfig(canvas_frame_id, width=event.width)
            
        self.scrollable_frame.bind("<Configure>", on_frame_configure)
        # O bind acima no frame interno é importante para atualizar quando o conteúdo muda
        # O bind abaixo no canvas ajuda a ajustar a largura inicial e em redimensionamentos
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_frame_id, width=e.width)) 

        # --- Adicionar Scroll do Mouse (Opcional, mas bom para UX) ---
        def _on_mousewheel(event):
            # Ajustar delta para diferentes OS (Windows usa delta, Linux button 4/5)
            if event.num == 4: # Linux scroll up
                 delta = -1
            elif event.num == 5: # Linux scroll down
                 delta = 1
            else: # Windows
                 delta = -int(event.delta / 120)
            canvas.yview_scroll(delta, "units")
            
        # Bind para Windows/macOS e Linux
        # Usar o container pai para pegar o evento mesmo se o mouse estiver sobre o padding
        left_frame_container.bind_all("<MouseWheel>", _on_mousewheel) # Windows
        left_frame_container.bind_all("<Button-4>", _on_mousewheel)   # Linux scroll up
        left_frame_container.bind_all("<Button-5>", _on_mousewheel)   # Linux scroll down

        # --- Painel Direito para Imagem (sem alterações) ---
        right_panel = ttk.Frame(root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ======================================================================
        # AGORA, ADICIONAR TODOS OS FRAMES DE CONTROLE AO self.scrollable_frame
        # ======================================================================

        # --- Frame de Configurações (no frame rolável) ---
        settings_frame = ttk.LabelFrame(self.scrollable_frame, text="Configurações", padding="10")
        settings_frame.pack(fill=tk.X, pady=5, anchor='n')
        ttk.Label(settings_frame, text="Largura:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.width_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(settings_frame, text="Altura:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.height_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.height_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(settings_frame, text="Semente:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.seed_var = tk.IntVar(value=random.randint(0, 100))
        self.seed_entry = ttk.Entry(settings_frame, textvariable=self.seed_var, width=10)
        self.seed_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Aleatória", command=self.randomize_seed).grid(row=1, column=2, padx=5, pady=5)

        # --- Frame de Processamento (CPU/GPU) (no frame rolável) ---
        self.proc_frame = ttk.LabelFrame(self.scrollable_frame, text="Processamento", padding="10") 
        self.proc_frame.pack(fill=tk.X, pady=5, anchor='n')
        self.processing_device = tk.StringVar(value="CPU")
        cpu_radio = ttk.Radiobutton(self.proc_frame, text="CPU", variable=self.processing_device, value="CPU", command=self._check_gpu_availability)
        cpu_radio.pack(side=tk.LEFT, padx=5)
        gpu_radio = ttk.Radiobutton(self.proc_frame, text="GPU (NVIDIA CUDA)", variable=self.processing_device, value="GPU", command=self._check_gpu_availability)
        gpu_radio.pack(side=tk.LEFT, padx=5)
        self.gpu_status_label = ttk.Label(self.proc_frame, text="")
        self.gpu_status_label.pack(side=tk.LEFT, padx=5)

        # --- Frame de Parâmetros do Ruído (no frame rolável) ---
        noise_frame = ttk.LabelFrame(self.scrollable_frame, text="Parâmetros do Ruído", padding="10")
        noise_frame.pack(fill=tk.X, pady=5, anchor='n')
        ttk.Label(noise_frame, text="Tipo:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.noise_type_var = tk.StringVar(value="Simplex")
        noise_combo = ttk.Combobox(noise_frame, textvariable=self.noise_type_var, values=["Simplex", "Perlin"], state="readonly", width=10)
        noise_combo.grid(row=0, column=1, padx=5, pady=5)
        noise_combo.bind('<<ComboboxSelected>>', self._noise_type_changed)
        ttk.Label(noise_frame, text="Escala Mundo:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.scale_var = tk.DoubleVar(value=10.0)
        self.scale_display_var = tk.StringVar()
        ttk.Scale(noise_frame, from_=1.0, to=50.0, orient=tk.HORIZONTAL, variable=self.scale_var, length=150, command=self._update_noise_labels).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.scale_display_var).grid(row=1, column=2, padx=5, pady=2)
        ttk.Label(noise_frame, text="Oitavas:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.octaves_var = tk.IntVar(value=6)
        ttk.Scale(noise_frame, from_=1, to=16, orient=tk.HORIZONTAL, variable=self.octaves_var, command=lambda v: self.octaves_var.set(int(float(v)))).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.octaves_var).grid(row=2, column=2, padx=5, pady=2)
        ttk.Label(noise_frame, text="Persistência:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.persistence_var = tk.DoubleVar(value=0.5)
        self.persistence_display_var = tk.StringVar()
        ttk.Scale(noise_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.persistence_var, command=self._update_noise_labels).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.persistence_display_var).grid(row=3, column=2, padx=5, pady=2)
        ttk.Label(noise_frame, text="Lacunaridade:").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        self.lacunarity_var = tk.DoubleVar(value=2.0)
        self.lacunarity_display_var = tk.StringVar()
        ttk.Scale(noise_frame, from_=1.0, to=4.0, orient=tk.HORIZONTAL, variable=self.lacunarity_var, command=self._update_noise_labels).grid(row=4, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.lacunarity_display_var).grid(row=4, column=2, padx=5, pady=2)

        # --- Frame de Pós-processamento (no frame rolável) ---
        post_proc_frame = ttk.LabelFrame(self.scrollable_frame, text="Pós-processamento", padding="10")
        post_proc_frame.pack(fill=tk.X, pady=5, anchor='n')
        self.terracing_var = tk.BooleanVar(value=False)
        self.terracing_check = ttk.Checkbutton(post_proc_frame, text="Habilitar Terraçamento", variable=self.terracing_var, command=self._toggle_terrace_levels)
        self.terracing_check.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.terrace_levels_label = ttk.Label(post_proc_frame, text="Níveis:")
        self.terrace_levels_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.terrace_levels_var = tk.IntVar(value=8)
        self.terrace_levels_scale = ttk.Scale(post_proc_frame, from_=2, to=500, orient=tk.HORIZONTAL, variable=self.terrace_levels_var, command=lambda v: self.terrace_levels_var.set(int(float(v))))
        self.terrace_levels_scale.grid(row=0, column=2, padx=5, pady=5)
        self.terrace_levels_display = ttk.Label(post_proc_frame, textvariable=self.terrace_levels_var)
        self.terrace_levels_display.grid(row=0, column=3, padx=5, pady=5)
        ttk.Separator(post_proc_frame, orient=tk.HORIZONTAL).grid(row=1, columnspan=4, sticky="ew", pady=10)
        self.warping_var = tk.BooleanVar(value=False)
        self.warping_check = ttk.Checkbutton(post_proc_frame, text="Habilitar Distorção", variable=self.warping_var, command=self._toggle_warping_controls)
        self.warping_check.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.warp_amp_label = ttk.Label(post_proc_frame, text="Amplitude:")
        self.warp_amp_label.grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.warp_amplitude_var = tk.DoubleVar(value=10.0)
        self.warp_amp_display_var = tk.StringVar()
        self.warp_amp_scale = ttk.Scale(post_proc_frame, from_=0.0, to=50.0, orient=tk.HORIZONTAL, variable=self.warp_amplitude_var, command=self._update_warp_labels)
        self.warp_amp_scale.grid(row=3, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW)
        self.warp_amp_display = ttk.Label(post_proc_frame, textvariable=self.warp_amp_display_var)
        self.warp_amp_display.grid(row=3, column=3, padx=5, pady=2)
        self.warp_freq_label = ttk.Label(post_proc_frame, text="Frequência:")
        self.warp_freq_label.grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        self.warp_frequency_var = tk.DoubleVar(value=1.0)
        self.warp_freq_display_var = tk.StringVar()
        self.warp_freq_scale = ttk.Scale(post_proc_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL, variable=self.warp_frequency_var, command=self._update_warp_labels)
        self.warp_freq_scale.grid(row=4, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW)
        self.warp_freq_display = ttk.Label(post_proc_frame, textvariable=self.warp_freq_display_var)
        self.warp_freq_display.grid(row=4, column=3, padx=5, pady=2)
        ttk.Separator(post_proc_frame, orient=tk.HORIZONTAL).grid(row=5, columnspan=4, sticky="ew", pady=10)
        self.island_mask_var = tk.BooleanVar(value=False)
        self.island_mask_check = ttk.Checkbutton(post_proc_frame, text="Criar Ilha (Máscara)", variable=self.island_mask_var, command=self._toggle_island_controls)
        self.island_mask_check.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.island_falloff_label = ttk.Label(post_proc_frame, text="Suavidade Borda:")
        self.island_falloff_label.grid(row=7, column=0, padx=5, pady=2, sticky=tk.W)
        self.island_falloff_var = tk.DoubleVar(value=0.8)
        self.island_falloff_display_var = tk.StringVar()
        self.island_falloff_scale = ttk.Scale(post_proc_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.island_falloff_var, command=self._update_island_labels)
        self.island_falloff_scale.grid(row=7, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW)
        self.island_falloff_display = ttk.Label(post_proc_frame, textvariable=self.island_falloff_display_var)
        self.island_falloff_display.grid(row=7, column=3, padx=5, pady=2)
        ttk.Separator(post_proc_frame, orient=tk.HORIZONTAL).grid(row=8, columnspan=4, sticky="ew", pady=10)
        self.remap_label = ttk.Label(post_proc_frame, text="Remapear (Potência):")
        self.remap_label.grid(row=9, column=0, padx=5, pady=2, sticky=tk.W)
        self.remap_exponent_var = tk.DoubleVar(value=1.0)
        self.remap_display_var = tk.StringVar()
        self.remap_scale = ttk.Scale(post_proc_frame, from_=0.1, to=4.0, orient=tk.HORIZONTAL, variable=self.remap_exponent_var, command=self._update_remap_label)
        self.remap_scale.grid(row=9, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW)
        self.remap_display = ttk.Label(post_proc_frame, textvariable=self.remap_display_var)
        self.remap_display.grid(row=9, column=3, padx=5, pady=2)
        ttk.Separator(post_proc_frame, orient=tk.HORIZONTAL).grid(row=10, columnspan=4, sticky="ew", pady=10)
        self.lakes_var = tk.BooleanVar(value=False)
        self.lakes_check = ttk.Checkbutton(post_proc_frame, text="Gerar Lagos (Nível do Mar)", variable=self.lakes_var, command=self._toggle_lakes_controls)
        self.lakes_check.grid(row=11, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)
        self.sea_level_label = ttk.Label(post_proc_frame, text="Nível Mar (0-255):")
        self.sea_level_label.grid(row=12, column=0, padx=5, pady=2, sticky=tk.W)
        self.sea_level_var = tk.IntVar(value=70)
        self.sea_level_scale = ttk.Scale(post_proc_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.sea_level_var, command=lambda v: self.sea_level_var.set(int(float(v))))
        self.sea_level_scale.grid(row=12, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW)
        self.sea_level_display = ttk.Label(post_proc_frame, textvariable=self.sea_level_var)
        self.sea_level_display.grid(row=12, column=3, padx=5, pady=2)

        # Montanhas Detalhadas (Ridged)
        ttk.Separator(post_proc_frame, orient=tk.HORIZONTAL).grid(row=13, columnspan=4, sticky="ew", pady=10)
        self.ridged_var = tk.BooleanVar(value=False)
        self.ridged_check = ttk.Checkbutton(post_proc_frame, text="Montanhas Detalhadas (Ridged)", variable=self.ridged_var)
        self.ridged_check.grid(row=14, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W)
        # Poderíamos adicionar controles para "nitidez" aqui se quiséssemos

        post_proc_frame.columnconfigure(1, weight=1)
        post_proc_frame.columnconfigure(2, weight=1)

        # --- Frame de Controles (no frame rolável) ---
        controls_frame = ttk.Frame(self.scrollable_frame, padding="10")
        controls_frame.pack(fill=tk.X, pady=5, anchor='n')
        self.generate_button = ttk.Button(controls_frame, text="Gerar Heightmap", command=self.generate_heightmap)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(controls_frame, text="Salvar Imagem", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        # Adicionar botão de Ajuda
        self.help_button = ttk.Button(controls_frame, text="Ajuda", command=self._show_help_window)
        self.help_button.pack(side=tk.RIGHT, padx=5) # Colocar à direita

        self.progress_bar = ttk.Progressbar(self.scrollable_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')

        self.image_label = ttk.Label(right_panel, text="A imagem gerada aparecerá aqui", anchor=tk.CENTER, borderwidth=1, relief="solid")
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.generated_image = None

        self._update_noise_labels()
        self._update_warp_labels()
        self._update_island_labels()
        self._update_remap_label()
        self._toggle_terrace_levels()
        self._toggle_warping_controls()
        self._toggle_island_controls()
        self._toggle_lakes_controls()
        self._check_gpu_availability()
        self._noise_type_changed()

    def _update_noise_labels(self, event=None):
        self.scale_display_var.set(f"{self.scale_var.get():.1f}")
        self.persistence_display_var.set(f"{self.persistence_var.get():.2f}")
        self.lacunarity_display_var.set(f"{self.lacunarity_var.get():.1f}")

    def randomize_seed(self):
        self.seed_var.set(random.randint(0, 99999))

    def _toggle_terrace_levels(self):
        """Habilita/Desabilita os controles de níveis de terraçamento."""
        # Checkbutton deve estar sempre clicável
        self.terracing_check.config(state=tk.NORMAL) 
        # Estado dos controles dependentes depende da variável
        label_scale_state = tk.NORMAL if self.terracing_var.get() else tk.DISABLED
        self.terrace_levels_label.config(state=label_scale_state)
        self.terrace_levels_scale.config(state=label_scale_state)
        self.terrace_levels_display.config(state=label_scale_state)

    def _toggle_warping_controls(self):
        """Habilita/Desabilita os controles de distorção de domínio."""
        # Checkbutton deve estar sempre clicável
        self.warping_check.config(state=tk.NORMAL) 
        # Estado dos controles dependentes depende da variável
        label_scale_state = tk.NORMAL if self.warping_var.get() else tk.DISABLED
        self.warp_amp_label.config(state=label_scale_state)
        self.warp_amp_scale.config(state=label_scale_state)
        self.warp_amp_display.config(state=label_scale_state)
        self.warp_freq_label.config(state=label_scale_state)
        self.warp_freq_scale.config(state=label_scale_state)
        self.warp_freq_display.config(state=label_scale_state)

    def _update_warp_labels(self, event=None):
        """Atualiza os StringVars dos labels de distorção com valores formatados."""
        self.warp_amp_display_var.set(f"{self.warp_amplitude_var.get():.1f}")
        self.warp_freq_display_var.set(f"{self.warp_frequency_var.get():.1f}")

    def _toggle_island_controls(self):
        """Habilita/Desabilita os controles da máscara de ilha."""
        state = tk.NORMAL if self.island_mask_var.get() else tk.DISABLED
        self.island_mask_check.config(state=tk.NORMAL) # Checkbutton sempre normal
        self.island_falloff_label.config(state=state)
        self.island_falloff_scale.config(state=state)
        self.island_falloff_display.config(state=state)

    def _update_island_labels(self, event=None):
        """Atualiza o label de suavidade da borda da ilha."""
        self.island_falloff_display_var.set(f"{self.island_falloff_var.get():.2f}")

    def _check_gpu_availability(self):
        """Verifica se Numba detecta uma GPU CUDA e atualiza a UI."""
        try:
            from numba import cuda
            if cuda.is_available():
                # Encontrou GPU CUDA
                devices = cuda.list_devices()
                if devices:
                    gpu_name = devices[0].name.decode('UTF-8')
                    self.gpu_status_label.config(text=f"GPU detectada: {gpu_name}", foreground="green")
                    # Habilita o Radiobutton da GPU (caso estivesse desabilitado)
                    for child in self.proc_frame.winfo_children():
                        if isinstance(child, ttk.Radiobutton) and child['value'] == "GPU":
                            child.config(state=tk.NORMAL)
                    return True # GPU disponível
                else:
                    self.gpu_status_label.config(text="Nenhuma GPU CUDA encontrada.", foreground="orange")
            else:
                self.gpu_status_label.config(text="CUDA não disponível (Verifique Toolkit/Driver).", foreground="red")
        except ImportError:
            self.gpu_status_label.config(text="Numba não encontrado?", foreground="red")
        except Exception as e:
            # Captura outros erros potenciais (ex: inicialização CUDA)
            self.gpu_status_label.config(text=f"Erro ao verificar GPU: {e}", foreground="red")
            print(f"Erro detalhado ao verificar GPU: {traceback.format_exc()}")

        # Se chegou aqui, GPU não está disponível ou houve erro
        # Desabilita o Radiobutton da GPU e força seleção para CPU
        for child in self.proc_frame.winfo_children():
             if isinstance(child, ttk.Radiobutton) and child['value'] == "GPU":
                 child.config(state=tk.DISABLED)
        self.processing_device.set("CPU")
        return False # GPU não disponível

    # --- Funções de Geração e Exibição --- 
    def generate_heightmap(self):
        """Função principal chamada pelo botão 'Gerar'. Decide qual método usar (CPU/GPU)."""
        device = self.processing_device.get()
        if device == "GPU":
            if not NUMBA_AVAILABLE:
                tk.messagebox.showerror("Erro", "A biblioteca Numba não está instalada. Não é possível usar a GPU.")
                return
            if not self._check_gpu_availability(): # Re-verifica caso algo tenha mudado
                tk.messagebox.showerror("Erro", "GPU não disponível ou erro na verificação. Usando CPU.")
                self.processing_device.set("CPU") # Força de volta para CPU
                self._generate_heightmap_cpu() # Tenta rodar na CPU
            else:
                self._generate_heightmap_gpu()
        else:
            self._generate_heightmap_cpu()

    def _generate_heightmap_cpu(self):
        """Gera o heightmap na CPU, escolhendo entre Simplex e Perlin."""
        start_time = time.time()
        try:
            self.progress_bar.config(mode='determinate')
            self.progress_bar.pack(fill=tk.X, pady=(10, 5), anchor='n')
            self.progress_bar['value'] = 0
            width = self.width_var.get()
            height = self.height_var.get()
            self.progress_bar['maximum'] = height
            self.root.update_idletasks()

            if width <= 0 or height <= 0:
                tk.messagebox.showerror("Erro", "Largura e Altura devem ser maiores que zero.")
                return
            
            scale = self.scale_var.get()
            octaves = self.octaves_var.get()
            persistence = self.persistence_var.get()
            lacunarity = self.lacunarity_var.get()
            seed = self.seed_var.get()
            warp_enabled = self.warping_var.get()
            warp_amplitude = self.warp_amplitude_var.get()
            warp_frequency = self.warp_frequency_var.get()
            noise_type = self.noise_type_var.get()
            remap_exponent = self.remap_exponent_var.get()
            island_mask_enabled = self.island_mask_var.get()
            island_falloff = self.island_falloff_var.get()
            terracing_enabled = self.terracing_var.get()
            terrace_levels = self.terrace_levels_var.get()
            lakes_enabled = self.lakes_var.get()
            sea_level = self.sea_level_var.get()
            ridged_enabled = self.ridged_var.get()

            print(f"Gerando heightmap (CPU - {noise_type}): {width}x{height}, seed={seed}, ...")
            
            # --- Geração do Ruído Base (Simplex ou Perlin) ---
            if noise_type == "Simplex":
                simplex = OpenSimplex(seed=seed)
                simplex_warp_x = OpenSimplex(seed=seed + 1)
                simplex_warp_y = OpenSimplex(seed=seed + 2)
                height_map = np.zeros((height, width), dtype=np.float64)
                update_interval = max(1, height // 100)

                for i in range(height):
                    for j in range(width):
                        nx_base = j / width
                        ny_base = i / height
                        nx_warped = nx_base
                        ny_warped = ny_base
                        if warp_enabled:
                            warp_x_noise = simplex_warp_x.noise2(nx_base * warp_frequency, ny_base * warp_frequency)
                            warp_y_noise = simplex_warp_y.noise2(nx_base * warp_frequency, ny_base * warp_frequency)
                            amplitude_factor = 100.0 
                            nx_warped += warp_x_noise * (warp_amplitude / amplitude_factor)
                            ny_warped += warp_y_noise * (warp_amplitude / amplitude_factor)
                        
                        # Calcula coordenadas base para o ruído usando a Escala Mundo
                        world_scale = self.scale_var.get()
                        nx_base_scaled = nx_warped * world_scale
                        ny_base_scaled = ny_warped * world_scale
                        
                        # Loop de Oitavas (usando coords escaladas)
                        total_noise = 0.0
                        frequency = 1.0
                        amplitude = 1.0

                        for k in range(octaves):
                            noise_val = simplex.noise2(nx_base_scaled * frequency, ny_base_scaled * frequency)
                            
                            total_noise += noise_val * amplitude
                            amplitude *= persistence
                            frequency *= lacunarity
                        # Agora total_noise contém o ruído fractal padrão
                        height_map[i, j] = total_noise
                    
                    self.progress_bar['value'] = i + 1
                    if (i + 1) % update_interval == 0:
                        self.root.update_idletasks()

            elif noise_type == "Perlin":
                if not NUMBA_AVAILABLE:
                    tk.messagebox.showerror("Erro", "Numba não encontrado. Não é possível gerar Perlin na CPU.")
                    return
                # Cria tabelas de permutação
                rng_main = np.random.RandomState(seed)
                perm_main = rng_main.permutation(256)
                perm_table_main = np.concatenate((perm_main, perm_main)).astype(np.int32)
                rng_warp_x = np.random.RandomState(seed + 1)
                perm_warp_x = rng_warp_x.permutation(256)
                perm_table_warp_x = np.concatenate((perm_warp_x, perm_warp_x)).astype(np.int32)
                rng_warp_y = np.random.RandomState(seed + 2)
                perm_warp_y = rng_warp_y.permutation(256)
                perm_table_warp_y = np.concatenate((perm_warp_y, perm_warp_y)).astype(np.int32)
                
                # Chama a função Numba @njit paralelizada
                world_scale = self.scale_var.get()
                height_map = perlin_cpu_octave_loop(
                    width, height, world_scale, octaves, persistence, lacunarity, seed,
                    warp_enabled, warp_amplitude, warp_frequency, 
                    perm_table_main, perm_table_warp_x, perm_table_warp_y
                )
                # Atualiza a barra de uma vez só, pois @njit bloqueia
                self.progress_bar['value'] = height
                self.root.update_idletasks()
            else:
                tk.messagebox.showerror("Erro", f"Tipo de ruído desconhecido: {noise_type}")
                return

            # --- Aplica Máscara de Ilha (CPU) ---
            if island_mask_enabled:
                print("Aplicando máscara de ilha...")
                center_x, center_y = width / 2, height / 2
                falloff = island_falloff # Usar a variável obtida no início
                x_coords = np.arange(width)
                y_coords = np.arange(height)
                xx, yy = np.meshgrid(x_coords, y_coords)
                dist_x = (xx - center_x) / center_x if center_x != 0 else np.zeros_like(xx)
                dist_y = (yy - center_y) / center_y if center_y != 0 else np.zeros_like(yy)
                distance = np.sqrt(dist_x**2 + dist_y**2)
                distance = np.clip(distance, 0.0, 1.0)
                mask = 1.0 - np.power(distance, 2.0 / max(0.01, falloff))
                mask = np.clip(mask, 0.0, 1.0)
                temp_min, temp_max = np.min(height_map), np.max(height_map)
                if temp_max > temp_min:
                    norm_temp = (height_map - temp_min) / (temp_max - temp_min)
                    masked_norm_temp = norm_temp * mask
                    height_map = masked_norm_temp * (temp_max - temp_min) + temp_min

            # --- Aplica Transformação Ridged (CPU) ---
            if ridged_enabled:
                print("Aplicando transformação Ridged...")
                # Aplicar 1.0 - abs(valor). Normalizar antes?
                # O ruído Perlin/Simplex geralmente está em [-A, A] onde A é próximo de 1
                # Vamos tentar aplicar diretamente:
                height_map = 1.0 - np.abs(height_map)
                # Opcional: Potência para acentuar (pode refazer o problema original)
                # height_map = np.power(height_map, 2.0) 

            # --- Normalização Temporária [0, 255] para rios/lagos ---
            # Precisamos dos valores em 0-255 para aplicar nível do mar e intensidade do rio
            temp_min, temp_max = np.min(height_map), np.max(height_map)
            if temp_max > temp_min:
                height_map_0_255 = (255 * (height_map - temp_min) / (temp_max - temp_min))
            else:
                height_map_0_255 = np.zeros((height, width))
            height_map_0_255 = height_map_0_255.astype(np.float64) # Manter como float para cálculos

            # --- Gerar Lagos Simples (Nível do Mar) ---
            if lakes_enabled:
                print(f"Aplicando nível do mar em {sea_level}...")
                height_map_0_255[height_map_0_255 < sea_level] = sea_level

            # --- Normalização Final [0, 1], Remapeamento e Escala [0, 255] ---
            # Normaliza o mapa modificado (com rios/lagos) para [0, 1]
            min_val = np.min(height_map_0_255)
            max_val = np.max(height_map_0_255)
            if max_val > min_val:
                 normalized_map_01 = (height_map_0_255 - min_val) / (max_val - min_val)
            else:
                 # Se ficou tudo plano (ex: só mar), usa 0
                 normalized_map_01 = np.zeros((height, width))
                 
            # Aplica Remapeamento (Potência)
            if remap_exponent != 1.0:
                 print(f"Aplicando remapeamento com expoente {remap_exponent:.2f}...")
                 normalized_map_01 = np.power(normalized_map_01, remap_exponent)
            
            # Escala final para [0, 255]
            final_map = (normalized_map_01 * 255).astype(np.uint8)

            # --- Aplica Terraçamento ---
            if terracing_enabled:
                if terrace_levels >= 2:
                    print(f"Aplicando terraçamento com {terrace_levels} níveis...")
                    level_size = 256 / terrace_levels 
                    terraced_map = (np.floor(final_map / level_size) * level_size).astype(np.uint8)
                    final_map = terraced_map # Atualiza o mapa final
                else:
                    print("Número de níveis de terraçamento inválido (< 2), ignorando.")
            
            height_map_uint8 = final_map # Nome final para consistência

            # Criação e Exibição da Imagem
            self.generated_image = Image.fromarray(height_map_uint8, 'L')
            self._display_image() 
            self.save_button.config(state=tk.NORMAL)
            end_time = time.time()
            print(f"Heightmap (CPU - {noise_type}) gerado com sucesso em {end_time - start_time:.2f} segundos!")
        except Exception as e:
            traceback.print_exc()
            tk.messagebox.showerror("Erro na Geração (CPU)", f"Ocorreu um erro: {e}")
            print(f"Erro durante geração CPU: {e}")
            self.save_button.config(state=tk.DISABLED)
        finally:
            self.progress_bar.pack_forget()
            self.root.update_idletasks()

    def _generate_heightmap_gpu(self):
        """Gera o heightmap usando Numba na GPU (CUDA) com Perlin Noise."""
        # ... (Verificações iniciais de Numba/CUDA) ...
        noise_type = self.noise_type_var.get()
        if noise_type == "Simplex":
            tk.messagebox.showwarning("Aviso GPU", "Geração Simplex não implementada na GPU. Usando Perlin.")
            # Poderia forçar self.noise_type_var.set("Perlin") aqui, mas a mensagem é suficiente
        
        start_time = time.time()
        try:
            # ... (Configuração da barra de progresso indeterminada) ...
            # ... (Obtenção de parâmetros: width, height, scale, etc... INCLUINDO remap_exponent, island_mask_enabled, island_falloff, terracing_enabled, terrace_levels) ...
            world_scale = self.scale_var.get() # Obter world_scale ANTES da chamada
            remap_exponent = self.remap_exponent_var.get()
            ridged_enabled = self.ridged_var.get()
            # ... (restante dos parâmetros)
            
            print(f"Gerando heightmap (GPU - Numba/CUDA Perlin): {width}x{height}, seed={seed}, ...")

            # --- Preparação para CUDA (incluindo novos parâmetros para o kernel) ---
            # ... (Criação das tabelas de permutação) ...
            rng_main = np.random.RandomState(seed)
            perm_main = rng_main.permutation(256)
            perm_table_main = np.concatenate((perm_main, perm_main)).astype(np.int32)
            rng_warp_x = np.random.RandomState(seed + 1)
            perm_warp_x = rng_warp_x.permutation(256)
            perm_table_warp_x = np.concatenate((perm_warp_x, perm_warp_x)).astype(np.int32)
            rng_warp_y = np.random.RandomState(seed + 2)
            perm_warp_y = rng_warp_y.permutation(256)
            perm_table_warp_y = np.concatenate((perm_warp_y, perm_warp_y)).astype(np.int32)
            height_map_gpu = cuda.device_array((height, width), dtype=np.float64)
            perm_table_main_gpu = cuda.to_device(perm_table_main)
            perm_table_warp_x_gpu = cuda.to_device(perm_table_warp_x)
            perm_table_warp_y_gpu = cuda.to_device(perm_table_warp_y)
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(width / threadsperblock[0])
            blockspergrid_y = math.ceil(height / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            # --- Lançamento do Kernel CUDA (v3, passando ridged_enabled) ---
            perlin_octave_kernel_v2[blockspergrid, threadsperblock](
                height_map_gpu, width, height, 
                world_scale, self.octaves_var.get(), self.persistence_var.get(), self.lacunarity_var.get(),
                self.seed_var.get(),
                self.warping_var.get(), self.warp_amplitude_var.get(), self.warp_frequency_var.get(),
                self.island_mask_var.get(), self.island_falloff_var.get(),
                perm_table_main_gpu, perm_table_warp_x_gpu, perm_table_warp_y_gpu
            )
            
            cuda.synchronize()
            height_map = height_map_gpu.copy_to_host()

            # --- Aplica Transformação Ridged (CPU, após cópia da GPU) ---
            if ridged_enabled:
                print("Aplicando transformação Ridged (pós-GPU)...")
                height_map = 1.0 - np.abs(height_map)
                # Opcional: height_map = np.power(height_map, 2.0)
                
            # --- Normalização Temporária [0, 255] ---
            temp_min, temp_max = np.min(height_map), np.max(height_map)
            if temp_max > temp_min:
                height_map_0_255 = (255 * (height_map - temp_min) / (temp_max - temp_min))
            else:
                height_map_0_255 = np.zeros((height, width))
            height_map_0_255 = height_map_0_255.astype(np.float64)
            
            # --- Gerar Lagos Simples (Nível do Mar) (CPU) ---
            if lakes_enabled:
                print(f"Aplicando nível do mar em {sea_level}...")
                height_map_0_255[height_map_0_255 < sea_level] = sea_level
                
            # --- Normalização Final [0, 1], Remapeamento e Escala [0, 255] (CPU) ---
            min_val = np.min(height_map_0_255)
            max_val = np.max(height_map_0_255)
            if max_val > min_val:
                 normalized_map_01 = (height_map_0_255 - min_val) / (max_val - min_val)
            else:
                 normalized_map_01 = np.zeros((height, width))
            if remap_exponent != 1.0:
                 print(f"Aplicando remapeamento com expoente {remap_exponent:.2f}...")
                 normalized_map_01 = np.power(normalized_map_01, remap_exponent)
            final_map = (normalized_map_01 * 255).astype(np.uint8)
            
            # --- Aplica Terraçamento (CPU) ---
            if terracing_enabled:
                if terrace_levels >= 2:
                    # ... (Código de terraçamento igual CPU, usando final_map) ...
                    print(f"Aplicando terraçamento com {terrace_levels} níveis...")
                    level_size = 256 / terrace_levels 
                    terraced_map = (np.floor(final_map / level_size) * level_size).astype(np.uint8)
                    final_map = terraced_map # Atualiza o mapa final
                else:
                    print("Número de níveis de terraçamento inválido (< 2), ignorando.")

            height_map_uint8 = final_map # Nome final

            # --- Criação e Exibição da Imagem ---
            self.generated_image = Image.fromarray(height_map_uint8, 'L')
            self._display_image() 
            self.save_button.config(state=tk.NORMAL)
            end_time = time.time()
            print(f"Heightmap (GPU Numba Perlin) gerado com sucesso em {end_time - start_time:.2f} segundos!")
        except Exception as e:
            traceback.print_exc()
            tk.messagebox.showerror("Erro na Geração (GPU)", f"Ocorreu um erro: {e}")
            print(f"Erro durante geração GPU: {e}")
            self.save_button.config(state=tk.DISABLED)
        finally:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.root.update_idletasks()
            
    def _display_image(self):
        """Função auxiliar para redimensionar e exibir a imagem gerada na label."""
        if self.generated_image is None: return

        width = self.width_var.get()
        height = self.height_var.get()
        display_width = self.image_label.winfo_width()
        if display_width < 10: display_width = 400 # Valor padrão inicial
        
        if width == 0: aspect_ratio = 1 
        else: aspect_ratio = height / width
        display_height = int(display_width * aspect_ratio)
        
        max_display_height = self.image_label.winfo_height() - 20 # Ajustar à altura da label
        if max_display_height < 10: max_display_height = 450 # Fallback
        if display_height > max_display_height:
            display_height = max_display_height
            if aspect_ratio == 0: display_width = max_display_height
            else: display_width = int(display_height / aspect_ratio)

        if display_width > 0 and display_height > 0:
            img_resized = self.generated_image.resize((display_width, display_height), Image.Resampling.NEAREST)
            self.photo_image = ImageTk.PhotoImage(img_resized)
        else:
             self.photo_image = ImageTk.PhotoImage(self.generated_image)

        self.image_label.config(image=self.photo_image, text="")
        self.image_label.image = self.photo_image

    def save_image(self):
        if self.generated_image is None:
            tk.messagebox.showwarning("Aviso", "Nenhuma imagem foi gerada para salvar.")
            return

        # Sugere um nome de arquivo inicial baseado nos parâmetros
        seed = self.seed_var.get()
        width = self.width_var.get()
        height = self.height_var.get()
        default_filename = f"heightmap_{width}x{height}_seed{seed}.png"

        filepath = filedialog.asksaveasfilename(
            title="Salvar Heightmap Como",
            initialfile=default_filename,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tif;*.tiff"),
                ("All files", "*.*")
            ])

        if not filepath:
            # Usuário cancelou a caixa de diálogo
            return

        try:
            # Salva a imagem original (alta resolução), não a redimensionada da tela
            self.generated_image.save(filepath)
            tk.messagebox.showinfo("Sucesso", f"Heightmap salvo com sucesso em: {filepath}")
            print(f"Imagem salva em: {filepath}")
        except Exception as e:
            tk.messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar a imagem.\nErro: {e}")
            print(f"Erro ao salvar imagem: {e}")

    def _noise_type_changed(self, event=None):
        """Chamado quando o tipo de ruído é alterado. Pode ajustar UI se necessário."""
        # Por enquanto, apenas verifica se GPU + Simplex está selecionado
        if self.processing_device.get() == "GPU" and self.noise_type_var.get() == "Simplex":
            if hasattr(self, 'gpu_status_label'): # Garante que label existe
                 self.gpu_status_label.config(text="GPU não suporta Simplex (usará Perlin).", foreground="orange")
        else:
            self._check_gpu_availability() # Volta a verificar status normal da GPU

    def _update_remap_label(self, event=None):
        """Atualiza o label de expoente de remapeamento."""
        self.remap_display_var.set(f"{self.remap_exponent_var.get():.2f}")

    def _toggle_lakes_controls(self):
        """Habilita/Desabilita os controles de lagos."""
        # Corrigido: state depende de self.lakes_var.get()
        label_scale_state = tk.NORMAL if self.lakes_var.get() else tk.DISABLED
        self.lakes_check.config(state=tk.NORMAL) # Checkbutton sempre normal
        self.sea_level_label.config(state=label_scale_state)
        self.sea_level_scale.config(state=label_scale_state)
        self.sea_level_display.config(state=label_scale_state)

    def _toggle_island_controls(self):
        """Habilita/Desabilita os controles da máscara de ilha."""
        state = tk.NORMAL if self.island_mask_var.get() else tk.DISABLED
        self.island_mask_check.config(state=tk.NORMAL) # Checkbutton sempre normal
        self.island_falloff_label.config(state=state)
        self.island_falloff_scale.config(state=state)
        self.island_falloff_display.config(state=state)

    def _show_help_window(self):
        """Cria e exibe uma janela Toplevel com instruções de uso."""
        help_win = tk.Toplevel(self.root)
        help_win.title("Ajuda - Gerador de Heightmap")
        help_win.geometry("700x600") # Aumentar um pouco
        help_win.transient(self.root)
        help_win.grab_set()

        text_frame = ttk.Frame(help_win, padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        help_text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, 
                                     padx=10, pady=10, font=("Segoe UI", 10), 
                                     borderwidth=0, relief=tk.FLAT) # Melhorar aparência
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=help_text_widget.yview)

        # --- Conteúdo da Ajuda (mesmo texto) ---
        help_content = """\
Instruções de Uso - Gerador de Heightmap

Este programa gera imagens de heightmap (mapas de altura) em escala de cinza, \
úteis para criar terrenos em jogos ou outras aplicações gráficas.\
Preto representa a menor altitude e branco a maior.

--- Opções Principais ---

*   Largura/Altura: Dimensões da imagem gerada em pixels.

*   Semente (Seed): Um número inicial para o gerador de números aleatórios. \
    A mesma semente com os mesmos parâmetros sempre produzirá a mesma imagem. \
    Use o botão "Aleatória" para gerar uma nova semente.

*   Processamento: Escolha entre "CPU" ou "GPU". A GPU (se disponível e \
    compatível com CUDA) é geralmente muito mais rápida, mas só funciona \
    com ruído Perlin. Uma GPU compatível e o CUDA Toolkit da NVIDIA \
    precisam estar instalados.

--- Parâmetros do Ruído ---

*   Tipo: Escolha entre "Simplex" (mais orgânico, disponível apenas na CPU) \
    ou "Perlin" (clássico, disponível em CPU e GPU).

*   Escala Mundo: Controla o "tamanho" das características do terreno. Valores \
    maiores criam montanhas/vales maiores (menos frequentes), valores menores \
    criam detalhes mais finos (mais frequentes).

*   Oitavas: Número de camadas de ruído sobrepostas. Mais oitavas adicionam \
    mais detalhes, mas aumentam o tempo de geração.

*   Persistência: Controla a influência das oitavas subsequentes. Valores \
    menores que 1 fazem com que cada oitava tenha menos influência que a \
    anterior (resultado mais suave). Valores maiores (ex: ~0.5) são comuns.

*   Lacunaridade: Controla o aumento da frequência (detalhe) a cada oitava. \
    Um valor comum é 2.0 (cada oitava tem o dobro da frequência da anterior).

--- Pós-processamento ---

Estes efeitos são aplicados APÓS a geração do ruído base.


*   Habilitar Terraçamento: Agrupa as alturas em níveis discretos (platôs).

    - Níveis: Quantidade de platôs desejada (2 a 99).

*   Habilitar Distorção: Distorce o terreno usando outra camada de ruído, \
    criando padrões mais sinuosos.

    - Amplitude: Intensidade da distorção.

    - Frequência: Nível de detalhe da distorção.

*   Criar Ilha (Máscara): Aplica um gradiente radial que abaixa as bordas \
    do mapa, criando uma forma de ilha.

    - Suavidade Borda: Controla a transição da terra para a "água". Valores \
    menores criam bordas mais íngremes, valores maiores criam transições suaves.

*   Remapear (Potência): Altera a distribuição das alturas. Um expoente > 1.0 \
    acentua picos e achata vales. Um expoente < 1.0 achata picos e acentua vales. \
    1.0 não tem efeito.

*   Gerar Lagos (Nível do Mar): Define um nível de altura. Tudo abaixo \
    desse nível é achatado para criar lagos/oceanos.

    - Nível Mar: Altura (0-255) onde a água começa.

*   Montanhas Detalhadas (Ridged): Modifica o ruído para criar cristas e \
    vales mais definidos, simulando montanhas erodidas.

--- Controles ---

*   Gerar Heightmap: Aplica os parâmetros e gera a imagem.

*   Salvar Imagem: Salva a imagem gerada (na resolução original) em formato \
    PNG, JPG, etc.

*   Ajuda: Abre esta janela.

"""
        
        # --- Configuração das Tags de Estilo ---
        help_text_widget.tag_configure("title", font=("Segoe UI", 14, "bold"), 
                                         justify=tk.CENTER, spacing3=15)
        help_text_widget.tag_configure("section", font=("Segoe UI", 11, "bold"), 
                                         spacing1=10, spacing3=8, lmargin1=0, lmargin2=0)
        help_text_widget.tag_configure("bullet_point", font=("Segoe UI", 10), 
                                         lmargin1=20, lmargin2=20, spacing1=5) # Indentação para bullets
        help_text_widget.tag_configure("sub_bullet_point", font=("Segoe UI", 10), 
                                         lmargin1=40, lmargin2=40, spacing1=3) # Indentação maior
        help_text_widget.tag_configure("normal_text", font=("Segoe UI", 10), 
                                          spacing1=5, spacing3=10)
        help_text_widget.tag_configure("option_name", font=("Segoe UI", 10, "bold"))
        
        # Insere o texto
        help_text_widget.insert(tk.END, help_content)
        
        # --- Aplica as Tags ---
        help_text_widget.tag_add("title", "1.0", "1.end") # Primeira linha é o título
        help_text_widget.tag_add("normal_text", "3.0", "5.end") # Texto introdutório

        # Aplica tag 'section' aos títulos --- ... ---
        start_index = "1.0"
        while True:
            # Procura por '---' no início da linha, possivelmente com espaços antes
            match_start = help_text_widget.search(r'^\s*---', start_index, stopindex=tk.END, regexp=True)
            if not match_start: break
            line_start = f"{match_start.split('.')[0]}.0"
            line_end = f"{match_start.split('.')[0]}.end"
            # Remove a tag normal_text para evitar conflito, caso exista
            help_text_widget.tag_remove("normal_text", line_start, line_end)
            help_text_widget.tag_add("section", line_start, line_end)
            start_index = line_end # Continua busca da linha seguinte

        # Aplica tags aos bullet points (*)
        start_index = "1.0"
        while True:
            # Procura por '*' no início da linha, possivelmente com espaços antes
            match_start = help_text_widget.search(r'^\s*\*', start_index, stopindex=tk.END, regexp=True)
            if not match_start: break
            line_start = f"{match_start.split('.')[0]}.0"
            line_end = f"{match_start.split('.')[0]}.end"
            help_text_widget.tag_remove("normal_text", line_start, line_end)
            help_text_widget.tag_add("bullet_point", line_start, line_end)

            # Aplica negrito ao nome da opção (texto depois de * até :)
            colon_pos = help_text_widget.search(":", line_start, stopindex=line_end)
            if colon_pos:
                # Encontra o início do texto útil após o marcador '*', ignorando espaços
                content_start_index = help_text_widget.search(r'\S', match_start + "+1c", stopindex=colon_pos, regexp=True)
                if content_start_index:
                     # Garante que o início seja após o marcador *
                     if help_text_widget.compare(content_start_index, ">", match_start):
                        help_text_widget.tag_add("option_name", content_start_index, colon_pos)

            start_index = line_end

        # Aplica tags aos sub-bullet points (-)
        start_index = "1.0"
        while True:
            # Procura por '-' no início da linha, possivelmente com espaços antes
            match_start = help_text_widget.search(r'^\s*-', start_index, stopindex=tk.END, regexp=True)
            if not match_start: break
            line_start = f"{match_start.split('.')[0]}.0"
            line_end = f"{match_start.split('.')[0]}.end"
            help_text_widget.tag_remove("normal_text", line_start, line_end)
            help_text_widget.tag_add("sub_bullet_point", line_start, line_end)

             # Aplica negrito ao nome da sub-opção (texto depois de - até :)
            colon_pos = help_text_widget.search(":", line_start, stopindex=line_end)
            if colon_pos:
                # Encontra o início do texto útil após o marcador '-', ignorando espaços
                content_start_index = help_text_widget.search(r'\S', match_start + "+1c", stopindex=colon_pos, regexp=True)
                if content_start_index:
                    # Garante que o início seja após o marcador -
                    if help_text_widget.compare(content_start_index, ">", match_start):
                        help_text_widget.tag_add("option_name", content_start_index, colon_pos)

            start_index = line_end

        # Aplica tag normal_text onde nenhuma outra tag principal foi aplicada
        # (Isto é um fallback, pode não ser estritamente necessário dependendo da cobertura)
        # help_text_widget.tag_raise("normal_text") # Pode não ser necessário

        # Desabilita edição e adiciona botão Fechar
        help_text_widget.config(state=tk.DISABLED)
        close_button = ttk.Button(help_win, text="Fechar", command=help_win.destroy)
        close_button.pack(pady=(5, 10))

        help_win.wait_window()

if __name__ == "__main__":
    root = tk.Tk()
    app = HeightmapGeneratorApp(root)
    root.mainloop() 