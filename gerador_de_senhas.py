"""
password_generator.py
Gerador de senhas seguro e configurável.
Uso como módulo:
    from password_generator import generate_password
    pw = generate_password(16, use_upper=True, use_digits=True, use_symbols=True)

Uso como CLI:
    python password_generator.py --length 20 --no-lower --symbols --copy
"""

import string
import secrets
import argparse
import sys
from typing import Optional

# Conjunto de caracteres ambíguos (opcional)
AMBIGUOUS = {'l', 'I', '1', 'O', '0', 'o'}

DEFAULT_SYMBOLS = '!@#$%^&*()-_=+[]{};:,.<>/?'  # você pode ajustar

def build_charset(use_lower: bool = True,
                  use_upper: bool = True,
                  use_digits: bool = True,
                  use_symbols: bool = True,
                  avoid_ambiguous: bool = False,
                  custom_symbols: Optional[str] = None) -> str:
    """Retorna a string com os caracteres a serem usados para gerar a senha."""
    parts = []
    if use_lower:
        parts.append(string.ascii_lowercase)
    if use_upper:
        parts.append(string.ascii_uppercase)
    if use_digits:
        parts.append(string.digits)
    if use_symbols:
        parts.append(custom_symbols if custom_symbols is not None else DEFAULT_SYMBOLS)

    charset = ''.join(parts)
    if not charset:
        raise ValueError("Charset vazio: pelo menos um tipo de caractere deve ser habilitado.")

    if avoid_ambiguous:
        charset = ''.join(ch for ch in charset if ch not in AMBIGUOUS)

    if not charset:
        raise ValueError("Charset ficou vazio após remover caracteres ambíguos.")

    return charset

def generate_password(length: int = 12,
                      use_lower: bool = True,
                      use_upper: bool = True,
                      use_digits: bool = True,
                      use_symbols: bool = False,
                      avoid_ambiguous: bool = False,
                      custom_symbols: Optional[str] = None) -> str:
    """Gera uma senha segura com as opções fornecidas.
    Usa `secrets.choice` para segurança criptográfica.
    Garante pelo menos um caractere de cada categoria habilitada (se possível).
    """
    if length <= 0:
        raise ValueError("Length must be positive")

    # Construir sets por categoria para garantir diversidade
    categories = []
    if use_lower:
        lower = string.ascii_lowercase
        if avoid_ambiguous:
            lower = ''.join(ch for ch in lower if ch not in AMBIGUOUS)
        categories.append(lower)
    if use_upper:
        upper = string.ascii_uppercase
        if avoid_ambiguous:
            upper = ''.join(ch for ch in upper if ch not in AMBIGUOUS)
        categories.append(upper)
    if use_digits:
        digits = string.digits
        if avoid_ambiguous:
            digits = ''.join(ch for ch in digits if ch not in AMBIGUOUS)
        categories.append(digits)
    if use_symbols:
        symbols = custom_symbols if custom_symbols is not None else DEFAULT_SYMBOLS
        if avoid_ambiguous:
            symbols = ''.join(ch for ch in symbols if ch not in AMBIGUOUS)
        categories.append(symbols)

    if not categories:
        raise ValueError("At least one character category must be enabled.")

    # Se length < number of enabled categories, não podemos garantir 1 por categoria — mas tentamos
    pwd_chars = []

    # Primeiro, garantir ao menos um caractere de cada categoria (se couber)
    for cat in categories:
        if len(pwd_chars) < length:
            pwd_chars.append(secrets.choice(cat))

    # Construir charset total e preencher o restante
    charset = ''.join(categories)
    while len(pwd_chars) < length:
        pwd_chars.append(secrets.choice(charset))

    # Misturar para não deixar os garantidos no início
    secrets.SystemRandom().shuffle(pwd_chars)
    return ''.join(pwd_chars)


# --- CLI ---
def main(argv=None):
    parser = argparse.ArgumentParser(description="Gerador de senhas seguro (Python).")
    parser.add_argument('--length', '-l', type=int, default=16, help='Tamanho da senha (inteiro).')
    parser.add_argument('--no-lower', action='store_true', help='Desabilitar letras minúsculas.')
    parser.add_argument('--no-upper', action='store_true', help='Desabilitar letras maiúsculas.')
    parser.add_argument('--no-digits', action='store_true', help='Desabilitar dígitos.')
    parser.add_argument('--symbols', action='store_true', help='Incluir símbolos.')
    parser.add_argument('--avoid-ambiguous', action='store_true', help='Evitar caracteres ambíguos como l, 1, O, 0.')
    parser.add_argument('--custom-symbols', type=str, help='Conjunto customizado de símbolos.')
    parser.add_argument('--copy', action='store_true', help='Copiar a senha para a área de transferência (requer pyperclip).')
    parser.add_argument('--count', type=int, default=1, help='Quantas senhas gerar.')

    args = parser.parse_args(argv)

    try:
        for _ in range(max(1, args.count)):
            pwd = generate_password(
                length=args.length,
                use_lower=not args.no_lower,
                use_upper=not args.no_upper,
                use_digits=not args.no_digits,
                use_symbols=args.symbols,
                avoid_ambiguous=args.avoid_ambiguous,
                custom_symbols=args.custom_symbols
            )
            print(pwd)
            if args.copy:
                try:
                    import pyperclip
                    pyperclip.copy(pwd)
                    print("(Senha copiada para a área de transferência.)")
                except Exception as e:
                    print("(Falha ao copiar para clipboard: {})".format(e), file=sys.stderr)
    except ValueError as exc:
        print("Erro:", exc, file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
