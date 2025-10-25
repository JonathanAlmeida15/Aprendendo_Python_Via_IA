#!/usr/bin/env python3
"""
currency_converter.py
Conversor de moedas usando a API pública exchangerate.host
Exemplo:
    python currency_converter.py --from USD --to BRL --amount 10
"""

import requests
import argparse
from decimal import Decimal, ROUND_HALF_UP
import sys

API_URL = "https://api.exchangerate.host/convert"

def convert_currency(amount: float, from_currency: str, to_currency: str) -> Decimal:
    """Converte uma quantia entre duas moedas via API exchangerate.host."""
    params = {
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "amount": amount
    }

    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Erro ao acessar a API: {e}")

    data = response.json()
    if not data.get("success"):
        raise ValueError("Erro na conversão: resposta inválida da API.")

    result = data.get("result")
    if result is None:
        raise ValueError("Não foi possível obter o resultado da conversão.")

    return Decimal(str(result)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def get_supported_currencies() -> list[str]:
    """Retorna a lista de moedas suportadas pela API."""
    try:
        response = requests.get("https://api.exchangerate.host/symbols", timeout=10)
        response.raise_for_status()
        data = response.json()
        return list(data["symbols"].keys())
    except Exception:
        raise RuntimeError("Falha ao obter lista de moedas suportadas.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Conversor de moedas via exchangerate.host API.")
    parser.add_argument("--from", "-f", dest="from_currency", required=True, help="Moeda de origem (ex: USD).")
    parser.add_argument("--to", "-t", dest="to_currency", required=True, help="Moeda de destino (ex: BRL).")
    parser.add_argument("--amount", "-a", type=float, required=True, help="Valor a converter.")
    parser.add_argument("--list", action="store_true", help="Listar moedas suportadas.")
    args = parser.parse_args(argv)

    if args.list:
        print("Buscando lista de moedas suportadas...")
        try:
            currencies = get_supported_currencies()
            print(", ".join(sorted(currencies)))
        except Exception as e:
            print(f"Erro: {e}", file=sys.stderr)
        return

    try:
        result = convert_currency(args.amount, args.from_currency, args.to_currency)
        print(f"{args.amount:.2f} {args.from_currency.upper()} = {result} {args.to_currency.upper()}")
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()