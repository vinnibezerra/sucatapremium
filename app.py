from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import io
import os
from datetime import datetime, timedelta
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Criar diretórios se não existirem
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models/', exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """Classe para serializar objetos NumPy em JSON"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif hasattr(obj, '__html__'):
            return str(obj)  # Para objetos que possuem representação HTML
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

class SucataAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.ibge_data = None
        
    def fetch_ibge_prices(self):
        """Captura preços de sucata do site do IBGE"""
        try:
            # Simulação de dados (substituir com parsing real quando disponível)
            self.ibge_data = {
                'data': datetime.now().strftime('%Y-%m-%d'),
                'sucata_ferrosa': 850.50,
                'sucata_nao_ferrosa': 1200.75,
                'ferro_gusa': 950.25,
                'aco_laminado': 1100.00
            }
            return self.ibge_data
            
        except Exception as e:
            print(f"Erro ao buscar dados do IBGE: {e}")
            return {
                'data': datetime.now().strftime('%Y-%m-%d'),
                'sucata_ferrosa': 850.50,
                'sucata_nao_ferrosa': 1200.75,
                'ferro_gusa': 950.25,
                'aco_laminado': 1100.00
            }
    
    def preprocess_data(self, df):
        """Pré-processa os dados da planilha"""
        try:
            # Converter colunas de data
            date_columns = ['data_compra']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Extrair features de data
            if 'data_compra' in df.columns:
                df['dia_semana_compra'] = df['data_compra'].dt.dayofweek
                df['mes_compra'] = df['data_compra'].dt.month
                df['trimestre_compra'] = df['data_compra'].dt.quarter
            
            # Codificar variáveis categóricas
            categorical_cols = ['tipo_sucata', 'fornecedor', 'regiao']
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Garantir que todas as colunas numéricas sejam float
            numeric_cols = ['peso', 'pureza', 'preco_compra', 'preco_venda']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Erro no pré-processamento: {e}")
            return df
    
    def train_model(self, df):
        """Treina modelo de machine learning"""
        try:
            # Verificar se temos dados suficientes
            if len(df) < 10:
                return {"error": "Dados insuficientes para treinamento (mínimo 10 registros)"}
            
            # Criar feature de lucro
            if all(col in df.columns for col in ['preco_venda', 'preco_compra']):
                df['lucro'] = df['preco_venda'] - df['preco_compra']
                target = 'lucro'
            else:
                return {"error": "Colunas de preço necessárias não encontradas"}
            
            # Definir features disponíveis
            available_features = []
            possible_features = ['peso', 'pureza', 'tipo_sucata', 'preco_compra', 'mes_compra', 'fornecedor']
            
            for feature in possible_features:
                if feature in df.columns and not df[feature].isnull().all():
                    available_features.append(feature)
            
            if len(available_features) < 2:
                return {"error": "Features insuficientes para treinamento"}
            
            # Preparar dados para treinamento
            X = df[available_features].fillna(df[available_features].mean())
            y = df[target]
            
            # Remover outliers extremos
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 5:
                return {"error": "Dados insuficientes após remoção de outliers"}
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Escalar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar modelo
            self.model = RandomForestRegressor(
                n_estimators=50,  # Reduzido para dados menores
                random_state=42,
                max_depth=10
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Avaliar modelo
            y_pred = self.model.predict(X_test_scaled)
            mse = float(mean_squared_error(y_test, y_pred))  # Converter para float
            r2 = float(r2_score(y_test, y_pred))  # Converter para float
            
            # Obter importância das features
            feature_importance = {}
            for i, feature in enumerate(available_features):
                feature_importance[feature] = float(self.model.feature_importances_[i])
            
            # Salvar modelo
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'features': available_features
            }, 'models/sucata_model.pkl')
            
            return {
                'mse': mse,
                'r2': r2,
                'features_importance': feature_importance,
                'features_used': available_features
            }
            
        except Exception as e:
            print(f"Erro no treinamento: {e}")
            return {"error": f"Erro no treinamento: {str(e)}"}
    
    def predict_optimal_trade(self, df, current_prices):
        """Prediz melhores pontos de compra e venda"""
        try:
            if self.model is None:
                return {"error": "Modelo não treinado"}
            
            # Preparar dados para predição
            df_processed = self.preprocess_data(df.copy())
            
            # Gerar cenários de preço
            scenarios = []
            price_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
            
            for multiplier in price_multipliers:
                scenario_df = df_processed.copy()
                scenario_df['preco_compra'] = scenario_df['preco_compra'] * multiplier
                
                # Usar as mesmas features do treinamento
                model_info = joblib.load('models/sucata_model.pkl')
                features_used = model_info['features']
                
                if all(feature in scenario_df.columns for feature in features_used):
                    X_scenario = scenario_df[features_used].fillna(scenario_df[features_used].mean())
                    X_scenario_scaled = self.scaler.transform(X_scenario)
                    
                    scenario_df['lucro_predito'] = self.model.predict(X_scenario_scaled)
                    
                    scenarios.append({
                        'multiplier': float(multiplier),
                        'avg_profit': float(scenario_df['lucro_predito'].mean()),
                        'max_profit': float(scenario_df['lucro_predito'].max()),
                        'min_profit': float(scenario_df['lucro_predito'].min())
                    })
            
            if not scenarios:
                return {"error": "Não foi possível gerar cenários"}
            
            # Encontrar melhor cenário
            best_scenario = max(scenarios, key=lambda x: x['avg_profit'])
            
            return {
                'scenarios': scenarios,
                'best_scenario': best_scenario,
                'recommendation': f"Multiplicador ideal de preço: {best_scenario['multiplier']:.2f}",
                'expected_profit': best_scenario['avg_profit']
            }
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return {"error": f"Erro na predição: {str(e)}"}
    
    def generate_visualizations(self, df):
        """Gera visualizações dos dados - VERSÃO CORRIGIDA"""
        try:
            visualizations = {}
            
            # Usar o DataFrame ORIGINAL para as visualizações
            df_viz = df.copy()
            
            print("Colunas disponíveis para visualização:", list(df_viz.columns))
            print("Amostra dos dados:")
            print(df_viz.head(3))
            
            # 1. Gráfico de Evolução de Preços ao Longo do Tempo
            if 'data_compra' in df_viz.columns and 'preco_compra' in df_viz.columns:
                try:
                    # Converter data_compra para datetime
                    df_viz['data_compra'] = pd.to_datetime(df_viz['data_compra'], errors='coerce')
                    
                    # Agrupar por data e calcular média
                    df_time = df_viz.groupby('data_compra').agg({
                        'preco_compra': 'mean',
                        'preco_venda': 'mean'
                    }).reset_index()
                    
                    print(f"Dados para gráfico temporal: {len(df_time)} registros")
                    
                    fig_time = go.Figure()
                    fig_time.add_trace(go.Scatter(x=df_time['data_compra'], y=df_time['preco_compra'],
                                                mode='lines+markers', name='Preço Compra', line=dict(color='blue')))
                    fig_time.add_trace(go.Scatter(x=df_time['data_compra'], y=df_time['preco_venda'],
                                                mode='lines+markers', name='Preço Venda', line=dict(color='green')))
                    fig_time.update_layout(title='Evolução dos Preços de Compra e Venda',
                                        xaxis_title='Data',
                                        yaxis_title='Preço (R$)',
                                        template='plotly_white',
                                        height=400)
                    
                    # Converter para JSON em vez de HTML
                    visualizations['price_evolution'] = fig_time.to_json()
                    print("✓ Gráfico de evolução gerado com sucesso")
                    
                except Exception as e:
                    print(f"✗ Erro no gráfico temporal: {e}")
            
             # 2. Gráfico Ultra-Seguro para Distribuição de Lucros
            if all(col in df_viz.columns for col in ['preco_compra', 'preco_venda']):
                try:
                    # Cálculo seguro - arredondar valores
                    df_viz['resultado'] = (df_viz['preco_venda'] - df_viz['preco_compra']).round(2)
        
                    # Análise segura da distribuição
                    lucro_stats = {
                        'positivos': len(df_viz[df_viz['resultado'] > 0]),
                        'negativos': len(df_viz[df_viz['resultado'] < 0]),
                        'neutros': len(df_viz[df_viz['resultado'] == 0]),
                        'faixa_min': df_viz['resultado'].min(),
                        'faixa_max': df_viz['resultado'].max()
                    }
        
                    # GRÁFICO SEGURO - FOCO EM TENDÊNCIAS, NÃO EM VALORES EXATOS
                    fig = go.Figure()
        
                    # Histograma principal
                    fig.add_trace(go.Histogram(
                        x=df_viz['resultado'],
                        name='Distribuição',
                        marker_color='#2E86AB',
                        opacity=0.7,
                        nbinsx=15,  # Número fixo de bins para segurança
                        hovertemplate='<b>Faixa de Resultado</b><br>Transações: %{y}<extra></extra>'
                    ))
        
                    # Linhas de referência discretas
                    fig.add_vline(x=0, line_dash="dash", line_color="red", 
                                 line_width=1, opacity=0.6)
        
                    # Área de destaque para lucros positivos
                    if lucro_stats['faixa_max'] > 0:
                        fig.add_vrect(x0=0, x1=lucro_stats['faixa_max'],
                                fillcolor="green", opacity=0.1, 
                                 annotation_text="Resultados Positivos", 
                                annotation_position="top right")
        
                    # Área de destaque para prejuízos
                    if lucro_stats['faixa_min'] < 0:
                        fig.add_vrect(x0=lucro_stats['faixa_min'], x1=0,
                                 fillcolor="red", opacity=0.1,
                                 annotation_text="Resultados Negativos",
                                 annotation_position="top left")
        
                    # Layout seguro e profissional
                    fig.update_layout(
                        title='📊 Distribuição de Resultados Financeiros',
                        xaxis_title='Resultado por Transação (R$)',
                        yaxis_title='Número de Transações',
                        template='plotly_white',
                        height=450,
                        showlegend=False,
                        font=dict(size=12),
                        margin=dict(l=50, r=50, t=80, b=50),
                        # Configurações de segurança
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.1)',
                            tickformat='.2f'  # Formato fixo para segurança
                        ),
                        yaxis=dict(
                            showgrid=False,
                            tickformat='d'  # Números inteiros
                        )
                    )
        
                    visualizations['profit_distribution'] = fig.to_json()
        
                except Exception as e:
                    print(f"❌ Erro seguro no gráfico de distribuição: {e}")

            # 3. Gráfico de Pizza: Lucro por Tipo de Sucata - CORES PROFISSIONAIS
            if 'tipo_sucata' in df_viz.columns and all(col in df_viz.columns for col in ['preco_compra', 'preco_venda']):
                try:
                    if 'lucro' not in df_viz.columns:
                        df_viz['lucro'] = df_viz['preco_venda'] - df_viz['preco_compra']
        
                    # Calcular lucro total por tipo de sucata
                    lucro_por_tipo = df_viz.groupby('tipo_sucata')['lucro'].sum().reset_index()
        
                    # PALETA PROFISSIONAL PARA DADOS FINANCEIROS
                    paletas_cores = {
                        'financeira': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A8EAE', '#F0F3BD', '#57CC99'],
                        'vibrante': ['#FF595E', '#FFCA3A', '#8AC926', '#1982C4', '#6A4C93', '#FF9F1C', '#2EC4B6', '#E71D36'],
                        'pastel': ['#A8E6CF', '#DCEDC1', '#FFD3B6', '#FFAAA5', '#FF8B94', '#D4A5A5', '#9C89B8', '#F0A6CA']
                        }
        
                    # Escolher uma paleta (recomendo 'financeira' ou 'vibrante')
                    paleta_escolhida = paletas_cores['financeira']
                    num_tipos = len(lucro_por_tipo)
                    cores_finais = paleta_escolhida[:num_tipos]
        
                    fig_pizza = px.pie(lucro_por_tipo, 
                          values='lucro', 
                          names='tipo_sucata',
                          title='📊 Distribuição do Lucro por Tipo de Sucata',
                          color_discrete_sequence=cores_finais)
        
                    fig_pizza.update_layout(
                            template='plotly_white', 
                            height=450,
                            legend=dict(
                                orientation="h", 
                                yanchor="bottom", 
                                y=-0.3, 
                                xanchor="center", 
                                x=0.5,
                                font=dict(size=11)
                                ),
                            title_x=0.5,
                            title_font=dict(size=16)
                            )
        
                    # Textos mais informativos
                    fig_pizza.update_traces(
                            textposition='inside' if num_tipos <= 5 else 'outside',
                            textinfo='percent+label',
                            textfont=dict(size=11, color='white' if num_tipos <= 5 else 'black'),
                            marker=dict(line=dict(color='white', width=2))
                            )
        
                    visualizations['profit_by_type'] = fig_pizza.to_json()
                    print(f"✓ Gráfico de pizza com {num_tipos} tipos de sucata gerado com sucesso")
        
                except Exception as e:
                    print(f"✗ Erro no gráfico de pizza por tipo: {e}")
            
            # 4. Gráfico Adaptativo Seguro com Linha Suavizada
            if all(col in df_viz.columns for col in ['preco_compra', 'preco_venda']):
                try:
                    n_transacoes = len(df_viz)
        
                    if n_transacoes <= 15:
                        df_viz['transacao'] = [f"T{i+1}" for i in range(n_transacoes)]
                        fig = px.bar(df_viz, x='transacao', y=['preco_compra', 'preco_venda'],
                        title=f'📊 Compra vs Venda ({n_transacoes} transações)',
                        barmode='group', labels={'value': 'Preço (R$)', 'variable': ''})
                        fig.update_layout(height=400)
                        # ... (código anterior para poucos dados) ...
                        pass
                    else:
                        opacity = max(0.2, 1.0 - (n_transacoes / 2000))

                        fig = px.scatter(df_viz, x='preco_compra', y='preco_venda',
                                        title='📈 Relação: Preço de Compra vs Preço de Venda',
                                        labels={'preco_compra': 'Preço de Compra (R$)', 
                                                'preco_venda': 'Preço de Venda (R$)'},
                                        color='lucro', opacity=opacity,
                                        color_continuous_scale='RdYlGn',
                                        color_continuous_midpoint=0)
                                        
            
                        # LINHA SUAVIZADA - OPÇÃO RECOMENDADA
                        max_val = max(df_viz[['preco_compra', 'preco_venda']].max().max(), 1) * 1.1
                        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                                mode='lines', 
                                                name='Compra = Venda',
                                                line=dict(color='blue',  # Cinza suave
                                                            dash='dot',  # Pontilhado sutil
                                                            width=2),
                                                opacity=0.5))
            
                        # Linha de tendência (mantém azul normal para contraste)
                        if n_transacoes > 10:
                            z = np.polyfit(df_viz['preco_compra'], df_viz['preco_venda'], 1)
                            p = np.poly1d(z)
                            fig.add_trace(go.Scatter(x=[0, max_val], y=[p(0), p(max_val)],
                                                    mode='lines', 
                                                    name='Tendência',
                                                    line=dict(color='green', width=1.2)))
            
                        fig.update_layout(
                                            template='plotly_white', 
                                            height=500,
                                            legend=dict(
                                                        orientation="h",      # Horizontal
                                                        yanchor="bottom",     # Âncora na base
                                                        y= -0.35,              # Acima do gráfico
                                                        xanchor="center",     # Centralizada
                                                        x=0.5,               # No centro
                                                        bgcolor='rgba(255,255,255,0.8)',
                                                        bordercolor='rgba(0,0,0,0.2)',
                                                        borderwidth=1,
                                                        font=dict(size=11)
                                                    ),
                                            margin=dict(t=50)  # Margem superior para caber a legenda
                                        )

                        
                    visualizations['scatter_plot'] = fig.to_json()
        
                except Exception as e:
                    print(f"❌ Erro no gráfico: {e}")
            
            # 5. Gráfico Ultra-Seguro para Fornecedores
            if 'fornecedor' in df_viz.columns and all(col in df_viz.columns for col in ['preco_compra', 'preco_venda']):
                try:
                    # Cálculos seguros
                    df_viz['lucro'] = df_viz['preco_venda'] - df_viz['preco_compra']
        
                    # Agrupar de forma anônima
                    supplier_stats = df_viz.groupby('fornecedor').agg({
                                                    'lucro': 'mean',
                                                    'preco_compra': 'count'
                    }).rename(columns={'preco_compra': 'volume'}).reset_index()
        
                    n_suppliers = len(supplier_stats)
        
                    # DECISÃO ADAPTATIVA SEM EXPOR NÚMEROS EXATOS
                    if n_suppliers <= 10:
                        # Barras horizontais para ranking claro
                        supplier_stats = supplier_stats.sort_values('lucro')
                        fig = px.bar(supplier_stats, y='fornecedor', x='lucro',
                                    title='📈 Rentabilidade por Parceiro Comercial',
                                    orientation='h',
                                    color='lucro', color_continuous_scale='RdYlGn',
                                    color_continuous_midpoint=0)
            
                        fig.update_layout(showlegend=False, height=400,
                                        xaxis_title='Rentabilidade (R$)',
                                        yaxis_title='')
            
                    else:
                        # Para muitos fornecedores: foco nos extremos
                        top_5 = supplier_stats.nlargest(5, 'lucro')
                        bottom_5 = supplier_stats.nsmallest(5, 'lucro')
                        highlights = pd.concat([top_5, bottom_5]).drop_duplicates()
            
                        fig = px.bar(highlights, y='fornecedor', x='lucro',
                                    title='🎯 Parceiros com Maior e Menor Rentabilidade',
                                    orientation='h',
                                    color='lucro', color_continuous_scale='RdYlGn')
            
                        fig.update_layout(showlegend=False, height=400)
        
                    # FORMATAÇÃO SEGURA FINAL
                    fig.update_traces(
                                    texttemplate='R$ %{x:.2f}',
                                    hovertemplate='<b>%{y}</b><br>Rentabilidade: R$ %{x:.2f}<extra></extra>'
                    )
        
                    # Remover qualquer informação sensível
                    fig.update_layout(
                                annotations=[],  # Limpar anotações
                                margin=dict(l=50, r=50, t=60, b=50)
                    )
        
                    visualizations['supplier_profit'] = fig.to_json()
        
                except Exception as e:
                    print(f"❌ Erro seguro no gráfico de fornecedores: {e}")
            
           # 6. Gráfico Ultra-Seguro para Distribuição por Tipo de Sucata
            if 'tipo_sucata' in df_viz.columns:
                try:
                    # Análise segura - sem expor dados sensíveis
                    tipo_analysis = df_viz['tipo_sucata'].value_counts().reset_index()
                    tipo_analysis.columns = ['categoria', 'volume']
                    tipo_analysis['participacao'] = (tipo_analysis['volume'] / len(df_viz)) * 100
        
                    n_categorias = len(tipo_analysis)
        
                    # PALETA PROFISSIONAL
                    cores_seguras = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', 
                                    '#6A8EAE', '#57CC99', '#FF9F1C', '#8AC926', '#1982C4']
        
                    # DECISÃO ADAPTATIVA INTELIGENTE
                    if n_categorias <= 5:
                        # Donut elegante para poucas categorias
                        fig = px.pie(tipo_analysis, values='volume', names='categoria',
                                    title='📊 Composição por Tipo de Material',
                                    hole=0.5,
                                    color_discrete_sequence=cores_seguras)
            
                        fig.update_traces(textinfo='percent+label', textposition='inside')
                        fig.update_layout(height=400, showlegend=False)
            
                    elif n_categorias <= 10:
                        # Barras horizontais para ranking claro
                        tipo_analysis = tipo_analysis.sort_values('volume')
                        fig = px.bar(tipo_analysis, y='categoria', x='volume',
                                    title='📈 Volume por Tipo de Material',
                                    orientation='h',
                                    color='volume', color_continuous_scale='Blues')
            
                        fig.update_traces(texttemplate='%{x} transações')
                        fig.update_layout(height=450, showlegend=False,
                                        xaxis_title='Volume de Transações')
            
                    else:
                        # Para muitas categorias: Top 10 + agrupamento
                        top_categorias = tipo_analysis.head(10)
                        if n_categorias > 10:
                            outros = pd.DataFrame({
                                'categoria': [f'Demais Materiais ({n_categorias - 10} categorias)'],
                                'volume': [tipo_analysis.tail(n_categorias - 10)['volume'].sum()],
                                'participacao': [tipo_analysis.tail(n_categorias - 10)['participacao'].sum()]
                            })
                            plot_data = pd.concat([top_categorias, outros])
                        else:
                            plot_data = top_categorias
            
                        plot_data = plot_data.sort_values('volume')
                        fig = px.bar(plot_data, y='categoria', x='volume',
                                    title='🏆 Principais Categorias de Material',
                                    orientation='h',
                                    color='volume', color_continuous_scale='Viridis')
            
                        fig.update_layout(height=500, showlegend=False)
        
                    # FORMATAÇÃO SEGURA FINAL
                    fig.update_layout(
                        template='plotly_white',
                        font=dict(size=12),
                        margin=dict(l=50, r=50, t=80, b=50),
                        # Remover informações sensíveis
                        annotations=[],
                        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                        yaxis=dict(showgrid=False)
                    )
        
                    # Tooltip discreto
                    fig.update_traces(
                        hovertemplate='<b>%{label}</b><br>Volume: %{value} transações<extra></extra>'
                    )
        
                    visualizations['type_distribution'] = fig.to_json()
        
                except Exception as e:
                    print(f"❌ Erro seguro no gráfico de tipos: {e}")
            
            print(f"✅ Total de visualizações geradas: {len(visualizations)}")
            
            return visualizations
            
        except Exception as e:
            print(f"❌ Erro geral na geração de visualizações: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_fallback_visualizations(self):
        """Gera visualizações básicas quando há erro"""
        try:
            # Criar um gráfico simples de exemplo
            fig = px.line(x=[1, 2, 3], y=[1, 3, 2], title='Exemplo de Gráfico')
            return {'fallback_viz': pio.to_html(fig, full_html=False)}
        except:
            return {}

analyzer = SucataAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nome de arquivo vazio'}), 400
        
        if file and file.filename.endswith(('.xlsx', '.xls')):
            # Salvar arquivo
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ler planilha - MANTER O DATAFRAME ORIGINAL
            df_original = pd.read_excel(filepath)
            df = df_original.copy()  # Trabalhar com cópia para pré-processamento
            
            # Verificar colunas obrigatórias
            required_cols = ['preco_compra', 'preco_venda']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return jsonify({'error': f'Colunas obrigatórias faltando: {missing_cols}'}), 400
            
            # Buscar dados do IBGE
            ibge_prices = analyzer.fetch_ibge_prices()
            
            # Pré-processar dados (apenas para o modelo)
            df_processed = analyzer.preprocess_data(df)
            
            # Treinar modelo
            training_results = analyzer.train_model(df_processed)
            
            # Fazer previsões
            predictions = analyzer.predict_optimal_trade(df_processed, ibge_prices)
            
            # Gerar visualizações com dados ORIGINAIS
            visualizations = analyzer.generate_visualizations(df_original)
        
            # Garantir que todas as visualizações sejam strings
            safe_visualizations = {}
            for key, value in visualizations.items():
                if value is not None:
                    safe_visualizations[key] = str(value)  # Converter para string)
            
            # Estatísticas básicas
            stats = {
                'total_records': int(len(df_original)),
                'avg_purchase_price': float(df_original['preco_compra'].mean()) if 'preco_compra' in df_original.columns else 0,
                'avg_sale_price': float(df_original['preco_venda'].mean()) if 'preco_venda' in df_original.columns else 0,
                'total_weight': float(df_original['peso'].sum()) if 'peso' in df_original.columns else 0,
                'unique_suppliers': int(df_original['fornecedor'].nunique()) if 'fornecedor' in df_original.columns else 0,
                'avg_profit': float((df_original['preco_venda'] - df_original['preco_compra']).mean()) if all(col in df_original.columns for col in ['preco_venda', 'preco_compra']) else 0
            }
            
            response_data = {
            'success': True,
            'stats': stats,
            'ibge_prices': ibge_prices,
            'training_results': training_results,
            'predictions': predictions,
            'visualizations': safe_visualizations,  # Usar versão segura
            'columns': list(df_original.columns),
            'sample_data': df_original.head(5).to_dict('records')
        }
            
            return app.response_class(
                response=json.dumps(response_data, cls=NumpyEncoder),
                status=200,
                mimetype='application/json'
            )
            
        else:
            return jsonify({'error': 'Formato de arquivo não suportado'}), 400
            
    except Exception as e:
        error_msg = f'Erro no processamento: {str(e)}'
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/download_template')
def download_template():
    """Fornece template de planilha para o usuário"""
    template_data = {
        'data_compra': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'tipo_sucata': ['FERRO', 'FERRO', 'CHAPARIA'],
        'peso': [1000, 1500, 1200],
        'pureza': [0.85, 0.92, 0.78],
        'preco_compra': [0.45, 0.50, 0.42],
        'preco_venda': [0.70, 0.75, 0.68],
        'fornecedor': ['Fornecedor A', 'Fornecedor B', 'Fornecedor A'],
        'regiao': ['Sudeste', 'Sudeste', 'Sudeste']
    }
    
    df_template = pd.DataFrame(template_data)
    output = io.BytesIO()
    df_template.to_excel(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='template_sucata.xlsx'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)