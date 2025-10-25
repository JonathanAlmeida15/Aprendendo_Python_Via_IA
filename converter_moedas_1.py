import requests

def convert_currency(amount, from_currency, to_currency):
    try:
        # Faz a requisição para a API Frankfurter
        url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
        response = requests.get(url)

        # Verifica se houve erro de conexão
        if response.status_code != 200:
            raise ValueError("Erro na requisição: status HTTP diferente de 200")

        data = response.json()
        print("DEBUG resposta API:", data)  # Para visualizar a resposta

        # Obtém o valor convertido do JSON retornado
        converted_amount = data["rates"].get(to_currency)
        if converted_amount is None:
            raise ValueError("Erro na conversão: moeda destino não encontrada")

        return converted_amount

    except Exception as e:
        print(f"Erro: {e}")
        return None


if __name__ == "__main__":
    print("=== Conversor de Moedas ===")
    amount = float(input("Digite o valor: "))
    from_currency = input("De (ex: USD): ").upper()
    to_currency = input("Para (ex: BRL): ").upper()

    result = convert_currency(amount, from_currency, to_currency)
    if result is not None:
        print(f"\n{amount} {from_currency} = {result:.2f} {to_currency}")
    else:
        print("\nFalha na conversão.")
