@echo off
title Reset Git Repository
color 0A

echo =======================================================
echo 🚀  REINICIALIZANDO REPOSITORIO GIT DO PROJETO
echo =======================================================
echo.

:: 1️⃣ Verifica se o Git está instalado
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERRO: Git nao encontrado no PATH.
    echo Instale o Git e tente novamente.
    pause
    exit /b
)

:: 2️⃣ Confirmação do usuário
set /p resposta="Tem certeza que deseja LIMPAR o repositorio e recriar do zero? (S/N): "
if /I "%resposta%" NEQ "S" (
    echo Operacao cancelada.
    pause
    exit /b
)

:: 3️⃣ Remove o histórico Git
echo.
echo 🧹 Removendo pasta .git...
rmdir /s /q .git

:: 4️⃣ Recria o repositório Git
echo.
echo 🔄 Iniciando novo repositório Git...
git init

:: 5️⃣ Cria .gitignore básico
echo.
echo 🛡️ Criando .gitignore padrão (Python + dados grandes)...
(
echo __pycache__/
echo *.pyc
echo *.pyo
echo *.pyd
echo .Python
echo env/
echo venv/
echo *.pkl
echo *.csv
echo *.zip
echo *.rar
echo *.h5
echo *.sqlite
echo *.db
) > .gitignore

:: 6️⃣ Solicita a URL do repositório remoto
set /p repoURL="Digite a URL do repositorio remoto (ex: https://github.com/usuario/repositorio.git): "
git remote add origin %repoURL%

:: 7️⃣ Adiciona e commita novamente
echo.
echo 🗂️ Adicionando arquivos...
git add .

echo.
git commit -m "Reinicializacao do projeto sem arquivos grandes"

:: 8️⃣ Faz o push forçado
echo.
echo ☁️ Enviando novo repositório para o GitHub...
git branch -M main
git push -f origin main

echo.
echo ✅ Processo concluido com sucesso!
echo O repositório foi limpo e reenviado.
echo =======================================================
pause
