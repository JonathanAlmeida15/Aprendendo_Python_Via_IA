import os
import shutil

# Dicionário que define as categorias e suas extensões
FILE_TYPES = {
    "Imagens": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
    "Documentos": [".pdf", ".doc", ".docx", ".txt", ".xls", ".xlsx", ".ppt", ".pptx"],
    "Áudios": [".mp3", ".wav", ".aac", ".ogg"],
    "Vídeos": [".mp4", ".avi", ".mov", ".mkv"],
    "Compactados": [".zip", ".rar", ".7z", ".tar", ".gz"],
}

def organize_folder(folder_path):
    """Organiza os arquivos da pasta informada em subpastas por tipo."""
    try:
        if not os.path.exists(folder_path):
            print("❌ A pasta informada não existe.")
            return

        # Cria subpastas, se ainda não existirem
        for category in FILE_TYPES.keys():
            os.makedirs(os.path.join(folder_path, category), exist_ok=True)
        os.makedirs(os.path.join(folder_path, "Outros"), exist_ok=True)

        # Percorre todos os arquivos na pasta
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Ignora diretórios
            if os.path.isdir(file_path):
                continue

            # Verifica a extensão do arquivo
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            # Encontra a categoria correspondente
            moved = False
            for category, extensions in FILE_TYPES.items():
                if ext in extensions:
                    dest_path = os.path.join(folder_path, category, filename)
                    shutil.move(file_path, dest_path)
                    print(f"📁 {filename} → {category}/")
                    moved = True
                    break

            # Se não encontrou categoria, vai pra "Outros"
            if not moved:
                dest_path = os.path.join(folder_path, "Outros", filename)
                shutil.move(file_path, dest_path)
                print(f"📦 {filename} → Outros/")

        print("\n✅ Organização concluída com sucesso!")

    except Exception as e:
        print(f"⚠️ Erro ao organizar: {e}")

if __name__ == "__main__":
    print("=== Organizador de Arquivos ===")
    folder = input("Digite o caminho da pasta que deseja organizar: ").strip()
    organize_folder(folder)
