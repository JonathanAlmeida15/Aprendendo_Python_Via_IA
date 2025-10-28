import requests

def convert_currency(amount, from_currency, to_currency):
    """
    Converte um valor monetário usando a API pública Frankfurter.
    Exemplo: 10 USD -> BRL
    """
    try:
        # Monta a URL com os parâmetros corretos
        url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"
        response = requests.get(url, timeout=10)

        # Debug para ver o que está acontecendo caso algo dê errado
        print("DEBUG URL:", response.url)
        print("DEBUG status:", response.status_code)

        # Verifica se a requisição foi bem-sucedida (HTTP 200)
        if response.status_code != 200:
            raise ValueError(f"Erro na requisição: status HTTP {response.status_code}")

        # Converte a resposta para JSON
        data = response.json()
        print("DEBUG resposta API:", data)

        # Pega a taxa de conversão e calcula o valor convertido
        rate = data["rates"].get(to_currency)
        if rate is None:
            raise ValueError("Moeda de destino não encontrada na resposta da API.")

        converted_amount = amount * rate
        return converted_amount, rate

    except Exception as e:
        print(f"Erro: {e}")
        return None, None


if __name__ == "__main__":
    print("=== Conversor de Moedas ===")

    try:
        amount = float(input("Digite o valor a converter: "))
        from_currency = input("De (ex: USD): ").upper()
        to_currency = input("Para (ex: BRL): ").upper()

        result, rate = convert_currency(amount, from_currency, to_currency)

        if result is not None:
            print(f"\n1 {from_currency} = {rate:.4f} {to_currency}")
            print(f"{amount} {from_currency} = {result:.2f} {to_currency}")
        else:
            print("\nFalha na conversão.")

    except ValueError:
        print("Por favor, digite um valor numérico válido.")
