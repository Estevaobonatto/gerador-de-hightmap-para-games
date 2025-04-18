import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageTk
from opensimplex import OpenSimplex
import random

class HeightmapGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gerador de Heightmap")
        self.root.geometry("600x750") # Tamanho inicial da janela

        # Estilo ttk
        style = ttk.Style()
        style.theme_use('clam') # Experimente outros temas como 'alt', 'default', 'classic'

        # --- Frame Principal --- 
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Frame de Configurações --- 
        settings_frame = ttk.LabelFrame(main_frame, text="Configurações", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)

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

        # --- Frame de Parâmetros do Ruído --- 
        noise_frame = ttk.LabelFrame(main_frame, text="Parâmetros do Ruído (Simplex)", padding="10")
        noise_frame.pack(fill=tk.X, pady=5)

        # Escala
        ttk.Label(noise_frame, text="Escala:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.scale_var = tk.DoubleVar(value=100.0)
        self.scale_display_var = tk.StringVar() # StringVar para exibição formatada
        ttk.Scale(noise_frame, from_=10.0, to=500.0, orient=tk.HORIZONTAL, variable=self.scale_var, length=200, command=self._update_noise_labels).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.scale_display_var).grid(row=0, column=2, padx=5, pady=2)

        # Oitavas
        ttk.Label(noise_frame, text="Oitavas:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.octaves_var = tk.IntVar(value=6)
        ttk.Scale(noise_frame, from_=1, to=16, orient=tk.HORIZONTAL, variable=self.octaves_var, command=lambda v: self.octaves_var.set(int(float(v)))).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.octaves_var).grid(row=1, column=2, padx=5, pady=2)

        # Persistência
        ttk.Label(noise_frame, text="Persistência:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.persistence_var = tk.DoubleVar(value=0.5)
        self.persistence_display_var = tk.StringVar() # StringVar para exibição formatada
        ttk.Scale(noise_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.persistence_var, command=self._update_noise_labels).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.persistence_display_var).grid(row=2, column=2, padx=5, pady=2)

        # Lacunaridade
        ttk.Label(noise_frame, text="Lacunaridade:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.lacunarity_var = tk.DoubleVar(value=2.0)
        self.lacunarity_display_var = tk.StringVar() # StringVar para exibição formatada
        ttk.Scale(noise_frame, from_=1.0, to=4.0, orient=tk.HORIZONTAL, variable=self.lacunarity_var, command=self._update_noise_labels).grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(noise_frame, textvariable=self.lacunarity_display_var).grid(row=3, column=2, padx=5, pady=2)

        # --- Frame de Pós-processamento --- 
        post_proc_frame = ttk.LabelFrame(main_frame, text="Pós-processamento", padding="10")
        post_proc_frame.pack(fill=tk.X, pady=5)

        # Terraçamento
        self.terracing_var = tk.BooleanVar(value=False)
        self.terracing_check = ttk.Checkbutton(post_proc_frame, text="Habilitar Terraçamento", variable=self.terracing_var, command=self._toggle_terrace_levels)
        self.terracing_check.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.terrace_levels_label = ttk.Label(post_proc_frame, text="Níveis:")
        self.terrace_levels_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.terrace_levels_var = tk.IntVar(value=8)
        self.terrace_levels_scale = ttk.Scale(post_proc_frame, from_=2, to=32, orient=tk.HORIZONTAL, variable=self.terrace_levels_var, command=lambda v: self.terrace_levels_var.set(int(float(v))))
        self.terrace_levels_scale.grid(row=0, column=2, padx=5, pady=5)
        self.terrace_levels_display = ttk.Label(post_proc_frame, textvariable=self.terrace_levels_var)
        self.terrace_levels_display.grid(row=0, column=3, padx=5, pady=5)

        # Inicialmente desabilita os controles de níveis
        self._toggle_terrace_levels()

        # --- Frame de Controles --- 
        controls_frame = ttk.Frame(main_frame, padding="10")
        controls_frame.pack(fill=tk.X, pady=5)

        self.generate_button = ttk.Button(controls_frame, text="Gerar Heightmap", command=self.generate_heightmap)
        self.generate_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(controls_frame, text="Salvar Imagem", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # --- Área de Visualização da Imagem --- 
        self.image_label = ttk.Label(main_frame, text="A imagem gerada aparecerá aqui", anchor=tk.CENTER)
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)
        self.generated_image = None # Para guardar a referência da imagem PIL

        # Chamar a atualização inicial dos labels de ruído
        self._update_noise_labels()

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
        self.terrace_levels_label.config(state=state)
        self.terrace_levels_scale.config(state=state)
        self.terrace_levels_display.config(state=state)

    # --- Funções de Geração e Exibição --- 
    def generate_heightmap(self):
        try:
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

            print(f"Gerando heightmap (OpenSimplex): {width}x{height}, seed={seed}, scale={scale:.1f}, octaves={octaves}, pers={persistence:.2f}, lac={lacunarity:.1f}")

            # Instancia OpenSimplex com a seed
            simplex = OpenSimplex(seed=seed)

            # Inicializa o array do mapa
            height_map = np.zeros((height, width))

            # --- Loop de Geração de Ruído com Oitavas ---
            for i in range(height):
                for j in range(width):
                    # Coordenadas normalizadas e escaladas
                    # Dividir por um valor (ex: 50) em vez de scale diretamente pode dar melhor controle inicial
                    # Ajuste este divisor conforme necessário para o efeito desejado
                    nx = j / width * (scale / 50.0)
                    ny = i / height * (scale / 50.0)

                    # Lógica de Oitavas (Fractal Noise)
                    total_noise = 0.0
                    frequency = 1.0
                    amplitude = 1.0
                    max_amplitude = 0.0 # Para normalização teórica, se necessária

                    for k in range(octaves):
                        # Gera ruído Simplex para esta oitava
                        # Os valores de noise2 estão no intervalo [-1, 1]
                        noise_val = simplex.noise2(nx * frequency, ny * frequency)

                        total_noise += noise_val * amplitude
                        max_amplitude += amplitude

                        # Atualiza amplitude e frequência para a próxima oitava
                        amplitude *= persistence
                        frequency *= lacunarity

                    # Guarda o valor final (pode ser necessário normalizar baseado em max_amplitude se desejado,
                    # mas normalizar o array final é mais robusto)
                    height_map[i][j] = total_noise

            # --- Normalização Final ---
            # Normaliza o array inteiro para o intervalo [0, 255]
            min_val = np.min(height_map)
            max_val = np.max(height_map)
            if max_val > min_val:
                 # Mapeia o intervalo [min_val, max_val] para [0, 255]
                 normalized_map = 255 * (height_map - min_val) / (max_val - min_val)
            else:
                 normalized_map = np.zeros((height, width)) # Mapa plano se não houver variação

            height_map_uint8 = normalized_map.astype(np.uint8)

            # --- Aplica Terraçamento (se habilitado) ---
            if self.terracing_var.get():
                num_levels = self.terrace_levels_var.get()
                if num_levels >= 2:
                    print(f"Aplicando terraçamento com {num_levels} níveis...")
                    # Calcula o tamanho de cada "degrau"
                    level_size = 256 / num_levels 
                    # Aplica a quantização (floor divide e multiplica)
                    terraced_map = (np.floor(height_map_uint8 / level_size) * level_size).astype(np.uint8)
                    # Opcional: Mapear para usar todo o intervalo 0-255 mais uniformemente
                    # factor = 255 / (level_size * (num_levels - 1)) if num_levels > 1 else 1
                    # terraced_map = (np.floor(height_map_uint8 / level_size) * level_size * factor).astype(np.uint8)
                    # Escolhi a versão mais simples acima, que deixa o último nível sem atingir 255 exato.
                    height_map_uint8 = terraced_map
                else:
                    print("Número de níveis de terraçamento inválido (< 2), ignorando.")

            # Cria a imagem PIL
            self.generated_image = Image.fromarray(height_map_uint8, 'L')

            # Exibe a imagem na interface
            # Redimensiona a imagem para caber na label se for muito grande (opcional)
            display_width = self.image_label.winfo_width()
            if display_width < 10: display_width = 400 # Valor padrão inicial
            
            # Evita divisão por zero se width for zero
            if width == 0: 
                aspect_ratio = 1 
            else:
                aspect_ratio = height / width

            display_height = int(display_width * aspect_ratio)
            
            # Garante que a imagem não exceda uma altura máxima também
            max_display_height = 450
            if display_height > max_display_height:
                display_height = max_display_height
                # Evita divisão por zero se aspect_ratio for zero
                if aspect_ratio == 0:
                    display_width = max_display_height
                else:
                    display_width = int(display_height / aspect_ratio)

            # Previne erro se display_width ou display_height for zero
            if display_width > 0 and display_height > 0:
                img_resized = self.generated_image.resize((display_width, display_height), Image.Resampling.NEAREST)
                self.photo_image = ImageTk.PhotoImage(img_resized)
            else:
                 # Se as dimensões forem inválidas, usa a imagem original
                 self.photo_image = ImageTk.PhotoImage(self.generated_image)

            self.image_label.config(image=self.photo_image, text="") # Remove texto placeholder
            self.image_label.image = self.photo_image # Guarda referência

            # Habilita o botão de salvar
            self.save_button.config(state=tk.NORMAL)
            print("Heightmap gerado com sucesso!")

        except Exception as e:
            import traceback
            traceback.print_exc() # Imprime traceback detalhado para depuração
            tk.messagebox.showerror("Erro na Geração", f"Ocorreu um erro: {e}")
            print(f"Erro durante geração: {e}")
            self.save_button.config(state=tk.DISABLED)

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
