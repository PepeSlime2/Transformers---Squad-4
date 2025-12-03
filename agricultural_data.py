"""
Dataset de Exemplo - Textos Agrícolas do Boletim 100 (SB100)
Instituto Agronômico de Campinas (IAC)

Este arquivo contém exemplos de textos do domínio agrícola que podem ser usados
para testes adicionais de benchmarking com modelos Transformer.

Fonte simulada: Capítulos sobre Citrus e Café do Boletim 100 do IAC
"""

# Textos de exemplo sobre Citrus
CITRUS_TEXTS = [
    """
    O cultivo de citros no Estado de São Paulo representa uma das principais 
    atividades agrícolas da região. As variedades de laranja mais cultivadas 
    incluem Pera, Valência e Hamlin, cada uma com características específicas 
    de produtividade e resistência a pragas.
    """,
    
    """
    A citricultura paulista enfrenta desafios importantes relacionados ao 
    greening (HLB - Huanglongbing), doença bacteriana transmitida por insetos 
    que afeta severamente a produção. O manejo integrado de pragas é essencial 
    para o controle efetivo.
    """,
    
    """
    O espaçamento ideal para plantio de citros varia conforme a variedade e 
    o porta-enxerto utilizado. Recomenda-se 7x4 metros para variedades de 
    porte médio em solos de fertilidade moderada, podendo ser ajustado 
    conforme condições específicas do terreno.
    """,
    
    """
    A nutrição adequada dos pomares cítricos requer atenção especial aos 
    macronutrientes nitrogênio, fósforo e potássio, além de micronutrientes 
    como zinco, manganês e boro. A análise foliar é ferramenta fundamental 
    para diagnóstico nutricional.
    """,
    
    """
    O controle de plantas daninhas em pomares de citros pode ser realizado 
    através de métodos mecânicos, químicos ou culturais. A roçada nas 
    entrelinhas e aplicação de herbicidas na linha de plantio são práticas 
    comuns na citricultura moderna.
    """
]

# Textos de exemplo sobre Café
COFFEE_TEXTS = [
    """
    A cafeicultura brasileira é responsável por aproximadamente um terço da 
    produção mundial de café. As principais espécies cultivadas são Coffea 
    arabica e Coffea canephora (robusta), sendo o arábica predominante nas 
    regiões de altitude.
    """,
    
    """
    O manejo da ferrugem do cafeeiro (Hemileia vastatrix) é crucial para 
    manutenção da produtividade. Pulverizações preventivas com fungicidas 
    cúpricos ou triazóis devem ser realizadas no início do período chuvoso, 
    antes do aparecimento dos sintomas.
    """,
    
    """
    A poda do cafeeiro visa renovar a estrutura produtiva da planta, eliminando 
    ramos velhos e estimulando a emissão de novos brotos. A poda por decote é 
    indicada para cafezais com mais de 20 anos ou quando a altura dificulta 
    as operações de colheita.
    """,
    
    """
    O armazenamento adequado do café beneficiado requer controle rigoroso de 
    umidade e temperatura. Grãos com umidade acima de 12% estão sujeitos ao 
    desenvolvimento de fungos e deterioração da qualidade. Armazéns bem 
    ventilados são essenciais.
    """,
    
    """
    A adubação do cafeeiro deve considerar a idade da planta, produtividade 
    esperada e análise de solo. Em lavouras adultas, aplicações parceladas 
    de nitrogênio durante o período chuvoso maximizam a eficiência de uso 
    do nutriente e reduzem perdas por lixiviação.
    """
]

# Perguntas e respostas para testes de QA
QA_EXAMPLES = [
    {
        "question": "Quais são as principais variedades de laranja cultivadas em São Paulo?",
        "context": CITRUS_TEXTS[0],
        "answer": "Pera, Valência e Hamlin"
    },
    {
        "question": "O que é HLB e como afeta a citricultura?",
        "context": CITRUS_TEXTS[1],
        "answer": "HLB (Huanglongbing) é uma doença bacteriana transmitida por insetos que afeta severamente a produção de citros."
    },
    {
        "question": "Qual o espaçamento recomendado para plantio de citros?",
        "context": CITRUS_TEXTS[2],
        "answer": "7x4 metros para variedades de porte médio em solos de fertilidade moderada."
    },
    {
        "question": "Quais são as principais espécies de café cultivadas no Brasil?",
        "context": COFFEE_TEXTS[0],
        "answer": "Coffea arabica e Coffea canephora (robusta), sendo o arábica predominante."
    },
    {
        "question": "Como deve ser realizado o armazenamento de café beneficiado?",
        "context": COFFEE_TEXTS[3],
        "answer": "Com controle rigoroso de umidade (abaixo de 12%) e temperatura, em armazéns bem ventilados."
    }
]

# Dados de classificação - categorias de tópicos agrícolas
CLASSIFICATION_EXAMPLES = [
    {"text": "O controle biológico de pragas utiliza organismos naturais...", "label": "Manejo de Pragas"},
    {"text": "A análise química do solo identifica deficiências nutricionais...", "label": "Fertilidade do Solo"},
    {"text": "Sistemas de irrigação por gotejamento economizam água...", "label": "Irrigação"},
    {"text": "A colheita mecanizada reduz custos de produção...", "label": "Mecanização"},
    {"text": "Variedades resistentes a doenças aumentam produtividade...", "label": "Melhoramento Genético"},
]

# Dados para sumarização
SUMMARIZATION_EXAMPLES = [
    {
        "text": """
        A agricultura de precisão utiliza tecnologias como GPS, sensores remotos 
        e análise de dados para otimizar o manejo de culturas. Através do 
        mapeamento detalhado de variabilidade espacial do solo e da planta, é 
        possível aplicar insumos de forma variável, reduzindo custos e impactos 
        ambientais. Drones equipados com câmeras multiespectrais permitem 
        monitoramento frequente das lavouras, identificando precocemente problemas 
        como estresse hídrico, deficiências nutricionais e ataques de pragas.
        """,
        "summary": "Agricultura de precisão usa GPS e sensores para otimizar manejo, aplicar insumos de forma variável e monitorar lavouras com drones."
    }
]


def get_agricultural_dataset(dataset_type="all"):
    """
    Retorna dataset de exemplos agrícolas
    
    Args:
        dataset_type: Tipo de dataset ('citrus', 'coffee', 'qa', 'classification', 'all')
    
    Returns:
        Dicionário ou lista com dados
    """
    if dataset_type == "citrus":
        return CITRUS_TEXTS
    elif dataset_type == "coffee":
        return COFFEE_TEXTS
    elif dataset_type == "qa":
        return QA_EXAMPLES
    elif dataset_type == "classification":
        return CLASSIFICATION_EXAMPLES
    elif dataset_type == "summarization":
        return SUMMARIZATION_EXAMPLES
    else:
        return {
            "citrus": CITRUS_TEXTS,
            "coffee": COFFEE_TEXTS,
            "qa": QA_EXAMPLES,
            "classification": CLASSIFICATION_EXAMPLES,
            "summarization": SUMMARIZATION_EXAMPLES
        }


if __name__ == "__main__":
    print("="*70)
    print(" Dataset de Exemplos Agrícolas - Projeto SB100")
    print("="*70)
    
    print("\n TEXTOS SOBRE CITRUS:")
    for i, text in enumerate(CITRUS_TEXTS, 1):
        print(f"\n{i}. {text.strip()[:100]}...")
    
    print("\n TEXTOS SOBRE CAFÉ:")
    for i, text in enumerate(COFFEE_TEXTS, 1):
        print(f"\n{i}. {text.strip()[:100]}...")
    
    print("\n EXEMPLOS DE PERGUNTAS E RESPOSTAS:")
    for i, qa in enumerate(QA_EXAMPLES, 1):
        print(f"\n{i}. Q: {qa['question']}")
        print(f"   R: {qa['answer']}")
    
    print("\n" + "="*70)
    print(f"Total de textos: {len(CITRUS_TEXTS) + len(COFFEE_TEXTS)}")
    print(f"Total de pares QA: {len(QA_EXAMPLES)}")
    print("="*70)
