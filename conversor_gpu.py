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

# ======================================================

class HeightmapGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gerador de Heightmap")
        # Ajustar geometria para layout lado a lado (largura maior)
        self.root.geometry("950x650") 

        # Estilo ttk
        style = ttk.Style()
        style.theme_use('clam')

        # --- Painel Esquerdo para Controles ---
        left_panel = ttk.Frame(root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), anchor='nw') # Fill Y, Anchor NorthWest

        # --- Painel Direito para Imagem ---
        right_panel = ttk.Frame(root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True) # Fill Both, Expand

        # --- Frame de Configurações (no painel esquerdo) ---
        settings_frame = ttk.LabelFrame(left_panel, text="Configurações", padding="10")
        settings_frame.pack(fill=tk.X, pady=5, anchor='n') # Fill X, Anchor North
        # Largura
        ttk.Label(settings_frame, text="Largura:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.width_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        # Altura
        ttk.Label(settings_frame, text="Altura:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.height_var = tk.IntVar(value=512)
        ttk.Entry(settings_frame, textvariable=self.height_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        # Semente (Seed)
        ttk.Label(settings_frame, text="Semente:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.seed_var = tk.IntVar(value=random.randint(0, 100))
        self.seed_entry = ttk.Entry(settings_frame, textvariable=self.seed_var, width=10)
        self.seed_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(settings_frame, text="Aleatória", command=self.randomize_seed).grid(row=1, column=2, padx=5, pady=5)

        # --- Frame de Processamento (CPU/GPU) ---
        self.proc_frame = ttk.LabelFrame(left_panel, text="Processamento", padding="10")
        self.proc_frame.pack(fill=tk.X, pady=5, anchor='n')

        self.processing_device = tk.StringVar(value="CPU") # Padrão para CPU
        cpu_radio = ttk.Radiobutton(self.proc_frame, text="CPU", variable=self.processing_device, value="CPU", command=self._check_gpu_availability)
        cpu_radio.pack(side=tk.LEFT, padx=5)
        gpu_radio = ttk.Radiobutton(self.proc_frame, text="GPU (NVIDIA CUDA)", variable=self.processing_device, value="GPU", command=self._check_gpu_availability)
        gpu_radio.pack(side=tk.LEFT, padx=5)
        self.gpu_status_label = ttk.Label(self.proc_frame, text="")
        self.gpu_status_label.pack(side=tk.LEFT, padx=5)

        # --- Frame de Parâmetros do Ruído (no painel esquerdo) ---
        noise_frame = ttk.LabelFrame(left_panel, text="Parâmetros do Ruído (Simplex)", padding="10")
        noise_frame.pack(fill=tk.X, pady=5, anchor='n') # Fill X, Anchor North
        # Escala
        ttk.Label(noise_frame, text="Escala:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.scale_var = tk.DoubleVar(value=100.0)
        self.scale_display_var = tk.StringVar()
        ttk.Scale(noise_frame, from_=10.0, to=500.0, orient=tk.HORIZONTAL, variable=self.scale_var, length=150, command=self._update_noise_labels).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.scale_display_var).grid(row=0, column=2, padx=5, pady=2)
        # Oitavas
        ttk.Label(noise_frame, text="Oitavas:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.octaves_var = tk.IntVar(value=6)
        ttk.Scale(noise_frame, from_=1, to=16, orient=tk.HORIZONTAL, variable=self.octaves_var, command=lambda v: self.octaves_var.set(int(float(v)))).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.octaves_var).grid(row=1, column=2, padx=5, pady=2)
        # Persistência
        ttk.Label(noise_frame, text="Persistência:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.persistence_var = tk.DoubleVar(value=0.5)
        self.persistence_display_var = tk.StringVar()
        ttk.Scale(noise_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.persistence_var, command=self._update_noise_labels).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.persistence_display_var).grid(row=2, column=2, padx=5, pady=2)
        # Lacunaridade
        ttk.Label(noise_frame, text="Lacunaridade:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.lacunarity_var = tk.DoubleVar(value=2.0)
        self.lacunarity_display_var = tk.StringVar()
        ttk.Scale(noise_frame, from_=1.0, to=4.0, orient=tk.HORIZONTAL, variable=self.lacunarity_var, command=self._update_noise_labels).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.lacunarity_display_var).grid(row=3, column=2, padx=5, pady=2)

        # --- Frame de Pós-processamento (no painel esquerdo) ---
        post_proc_frame = ttk.LabelFrame(left_panel, text="Pós-processamento", padding="10")
        post_proc_frame.pack(fill=tk.X, pady=5, anchor='n') # Fill X, Anchor North
        # Terraçamento
        self.terracing_var = tk.BooleanVar(value=False)
        self.terracing_check = ttk.Checkbutton(post_proc_frame, text="Habilitar Terraçamento", variable=self.terracing_var, command=self._toggle_terrace_levels)
        self.terracing_check.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.terrace_levels_label = ttk.Label(post_proc_frame, text="Níveis:")
        self.terrace_levels_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.terrace_levels_var = tk.IntVar(value=8)
        self.terrace_levels_scale = ttk.Scale(post_proc_frame, from_=2, to=256, orient=tk.HORIZONTAL, variable=self.terrace_levels_var, command=lambda v: self.terrace_levels_var.set(int(float(v))))
        self.terrace_levels_scale.grid(row=0, column=2, padx=5, pady=5)
        self.terrace_levels_display = ttk.Label(post_proc_frame, textvariable=self.terrace_levels_var)
        self.terrace_levels_display.grid(row=0, column=3, padx=5, pady=5)
        # Distorção de Domínio
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
        # Configura colunas para expandir dentro do post_proc_frame
        post_proc_frame.columnconfigure(1, weight=1)
        post_proc_frame.columnconfigure(2, weight=1)

        # --- Frame de Controles (no painel esquerdo, no final) ---
        controls_frame = ttk.Frame(left_panel, padding="10")
        controls_frame.pack(fill=tk.X, pady=5, anchor='n') # Fill X, Anchor North
        self.generate_button = ttk.Button(controls_frame, text="Gerar Heightmap", command=self.generate_heightmap)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(controls_frame, text="Salvar Imagem", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # --- Barra de Progresso (abaixo dos controles) ---
        self.progress_bar = ttk.Progressbar(left_panel, orient=tk.HORIZONTAL, length=200, mode='determinate')
        # Empacotar abaixo dos controles, mas não mostrar inicialmente
        # Usaremos pack_forget() e pack() para mostrar/ocultar
        # self.progress_bar.pack(fill=tk.X, pady=(5, 0), anchor='n')

        # --- Área de Visualização da Imagem (no painel direito) ---
        self.image_label = ttk.Label(right_panel, text="A imagem gerada aparecerá aqui", anchor=tk.CENTER, borderwidth=1, relief="solid") # Adiciona borda para visualização
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.generated_image = None

        # Chamar a atualização inicial dos labels e estados
        self._update_noise_labels()
        self._update_warp_labels()
        self._toggle_terrace_levels()
        self._toggle_warping_controls()
        self._check_gpu_availability() # Verifica disponibilidade da GPU na inicialização

    def _update_noise_labels(self, event=None): # Adiciona event=None para aceitar o argumento do Scale
        """Atualiza os StringVars dos labels de ruído com valores formatados."""
        self.scale_display_var.set(f"{self.scale_var.get():.1f}")
        self.persistence_display_var.set(f"{self.persistence_var.get():.2f}")
        self.lacunarity_display_var.set(f"{self.lacunarity_var.get():.1f}")

    def randomize_seed(self):
        self.seed_var.set(random.randint(0, 99999))

    def _toggle_terrace_levels(self):
        """Habilita/Desabilita os controles de níveis de terraçamento."""
        state = tk.NORMAL if self.terracing_var.get() else tk.DISABLED
        self.terracing_check.config(state=tk.NORMAL)
        label_scale_state = tk.NORMAL if self.terracing_var.get() else tk.DISABLED
        self.terrace_levels_label.config(state=label_scale_state)
        self.terrace_levels_scale.config(state=label_scale_state)
        self.terrace_levels_display.config(state=label_scale_state)

    def _toggle_warping_controls(self):
        """Habilita/Desabilita os controles de distorção de domínio."""
        state = tk.NORMAL if self.warping_var.get() else tk.DISABLED
        self.warping_check.config(state=tk.NORMAL)
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
        """Gera o heightmap usando OpenSimplex na CPU (código original)."""
        start_time = time.time() # Medir tempo
        try:
            # --- Início: Configurar e Mostrar Barra de Progresso (Determinada) ---
            self.progress_bar.config(mode='determinate') # Garantir modo determinado
            self.progress_bar.pack(fill=tk.X, pady=(10, 5), anchor='n') 
            self.progress_bar['value'] = 0
            width = self.width_var.get()
            height = self.height_var.get()
            self.progress_bar['maximum'] = height
            self.root.update_idletasks()

            # ... (Restante do código da função CPU original: obter params, loops, etc.) ...
            # Certifique-se que todo o código da antiga generate_heightmap esteja aqui
            if width <= 0 or height <= 0:
                tk.messagebox.showerror("Erro", "Largura e Altura devem ser maiores que zero.")
                return # Não precisa mais do pack_forget aqui por causa do finally
            
            scale = self.scale_var.get()
            octaves = self.octaves_var.get()
            persistence = self.persistence_var.get()
            lacunarity = self.lacunarity_var.get()
            seed = self.seed_var.get()
            warp_enabled = self.warping_var.get()
            warp_amplitude = self.warp_amplitude_var.get()
            warp_frequency = self.warp_frequency_var.get()

            print(f"Gerando heightmap (CPU - OpenSimplex): {width}x{height}, seed={seed}, scale={scale:.1f}, octaves={octaves}, pers={persistence:.2f}, lac={lacunarity:.1f}")
            if warp_enabled:
                print(f"  -> Distorção de Domínio: Habilitada (Amp: {warp_amplitude:.1f}, Freq: {warp_frequency:.1f})")

            simplex = OpenSimplex(seed=seed)
            simplex_warp_x = OpenSimplex(seed=seed + 1)
            simplex_warp_y = OpenSimplex(seed=seed + 2)
            height_map = np.zeros((height, width))
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
                    
                    nx_final = nx_warped * (scale / 50.0)
                    ny_final = ny_warped * (scale / 50.0)
                    total_noise = 0.0
                    frequency = 1.0
                    amplitude = 1.0
                    for k in range(octaves):
                        noise_val = simplex.noise2(nx_final * frequency, ny_final * frequency)
                        total_noise += noise_val * amplitude
                        amplitude *= persistence
                        frequency *= lacunarity
                    height_map[i][j] = total_noise
                
                self.progress_bar['value'] = i + 1
                if (i + 1) % update_interval == 0:
                    self.root.update_idletasks()

            # Normalização e Pós-processamento
            min_val = np.min(height_map)
            max_val = np.max(height_map)
            if max_val > min_val:
                 normalized_map = 255 * (height_map - min_val) / (max_val - min_val)
            else:
                 normalized_map = np.zeros((height, width))
            height_map_uint8 = normalized_map.astype(np.uint8)

            if self.terracing_var.get():
                num_levels = self.terrace_levels_var.get()
                if num_levels >= 2:
                    print(f"Aplicando terraçamento com {num_levels} níveis...")
                    level_size = 256 / num_levels 
                    terraced_map = (np.floor(height_map_uint8 / level_size) * level_size).astype(np.uint8)
                    height_map_uint8 = terraced_map
                else:
                    print("Número de níveis de terraçamento inválido (< 2), ignorando.")

            # Criação e Exibição da Imagem
            self.generated_image = Image.fromarray(height_map_uint8, 'L')
            self._display_image() # Chama função auxiliar para exibir
            self.save_button.config(state=tk.NORMAL)
            end_time = time.time()
            print(f"Heightmap (CPU) gerado com sucesso em {end_time - start_time:.2f} segundos!")

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
        if not NUMBA_AVAILABLE or not cuda.is_available():
             tk.messagebox.showerror("Erro GPU", "Numba/CUDA não está disponível.")
             return
        
        start_time = time.time()
        try:
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.pack(fill=tk.X, pady=(10, 5), anchor='n')
            self.progress_bar.start(10)
            self.root.update_idletasks()

            width = self.width_var.get()
            height = self.height_var.get()
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
            print(f"Gerando heightmap (GPU - Numba/CUDA Perlin): {width}x{height}, seed={seed}, ...")

            # --- Preparação para CUDA ---
            # Cria tabelas de permutação baseadas na seed (na CPU, depois copia para GPU)
            # Usaremos tabelas separadas para o ruído principal e para o warping
            rng_main = np.random.RandomState(seed)
            perm_main = rng_main.permutation(256)
            perm_table_main = np.concatenate((perm_main, perm_main)).astype(np.int32) # Tabela duplicada [0..511]
            
            rng_warp_x = np.random.RandomState(seed + 1)
            perm_warp_x = rng_warp_x.permutation(256)
            perm_table_warp_x = np.concatenate((perm_warp_x, perm_warp_x)).astype(np.int32)
            
            rng_warp_y = np.random.RandomState(seed + 2)
            perm_warp_y = rng_warp_y.permutation(256)
            perm_table_warp_y = np.concatenate((perm_warp_y, perm_warp_y)).astype(np.int32)

            # Aloca array de saída na GPU
            height_map_gpu = cuda.device_array((height, width), dtype=np.float64)
            # Copia tabelas de permutação para a GPU
            perm_table_main_gpu = cuda.to_device(perm_table_main)
            perm_table_warp_x_gpu = cuda.to_device(perm_table_warp_x)
            perm_table_warp_y_gpu = cuda.to_device(perm_table_warp_y)

            # Define configuração do Grid CUDA
            threadsperblock = (16, 16) # Tamanho comum de bloco 2D
            blockspergrid_x = math.ceil(width / threadsperblock[0])
            blockspergrid_y = math.ceil(height / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            # --- Lançamento do Kernel CUDA ---
            perlin_octave_kernel[blockspergrid, threadsperblock](
                height_map_gpu, width, height, 
                scale, octaves, persistence, lacunarity, 
                seed, # Seed pode não ser diretamente usada no kernel Perlin se a tabela já é baseada nela, mas passamos caso precise
                warp_enabled, warp_amplitude, warp_frequency,
                perm_table_main_gpu, perm_table_warp_x_gpu, perm_table_warp_y_gpu
            )
            
            # --- Sincronizar e Copiar Resultado ---
            cuda.synchronize() # Espera a GPU terminar
            height_map = height_map_gpu.copy_to_host() # Copia resultado de volta para a CPU

            # --- Normalização e Pós-processamento (CPU) ---
            # (Código existente de normalização e terraçamento permanece aqui)
            min_val = np.min(height_map)
            max_val = np.max(height_map)
            if max_val > min_val:
                 normalized_map = 255 * (height_map - min_val) / (max_val - min_val)
            else:
                 normalized_map = np.zeros((height, width))
            height_map_uint8 = normalized_map.astype(np.uint8)
            
            if self.terracing_var.get():
                num_levels = self.terrace_levels_var.get()
                if num_levels >= 2:
                    print(f"Aplicando terraçamento com {num_levels} níveis...")
                    level_size = 256 / num_levels 
                    terraced_map = (np.floor(height_map_uint8 / level_size) * level_size).astype(np.uint8)
                    height_map_uint8 = terraced_map
                else:
                    print("Número de níveis de terraçamento inválido (< 2), ignorando.")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = HeightmapGeneratorApp(root)
    root.mainloop() 