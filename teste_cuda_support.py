 # check_cuda.py
import sys

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
try:
    print("Tentando importar Numba...")
    from numba import cuda, __version__ as numba_version
    print(f"Numba versão: {numba_version}")

    print("\nVerificando disponibilidade CUDA...")
    is_cuda_available = cuda.is_available()
    print(f"CUDA disponível (cuda.is_available()): {is_cuda_available}")

    if is_cuda_available:
        print("\nListando dispositivos CUDA...")
        devices = cuda.list_devices()
        if devices:
            for i, device in enumerate(devices):
                try:
                    print(f"  Dispositivo {i}: {device.name.decode('UTF-8')}")
                except Exception as e:
                    print(f"  Erro ao obter nome do Dispositivo {i}: {e}")
            else:
                print("  Nenhum dispositivo CUDA encontrado por list_devices().")

            try:
                print("\nObtendo versão do CUDA Runtime linkado pelo Numba...")
                runtime_version = cuda.runtime.get_version()
                print(f"  Versão do CUDA Runtime (major.minor): {runtime_version[0]}.{runtime_version[1]}")
            except Exception as e:
                print(f"  Erro ao obter versão do CUDA Runtime: {e}")

        else:
             print("\nTentando obter versão do Driver NVIDIA (pode falhar se CUDA não inicializar)...")
             try:
                 driver_version = cuda.cudadrv.driver.get_version()
                 # A versão é retornada como um inteiro (ex: 11070 para 11.7). Precisamos formatar.
                 major = driver_version // 1000
                 minor = (driver_version % 1000) // 10
                 print(f"  Versão do Driver NVIDIA detectada pelo Numba: {major}.{minor} (raw: {driver_version})")
             except Exception as e:
                 print(f"  Não foi possível obter a versão do Driver NVIDIA via Numba: {e}")

except ImportError:
    print("\nERRO: A biblioteca Numba não foi encontrada ou não pôde ser importada.")
except Exception as e:
    import traceback
    print(f"\nERRO INESPERADO: {e}")
    traceback.print_exc()

    print("\n--- Verificação Concluída ---")
    input("Pressione Enter para sair...") # Mantém a janela aberta no Windows