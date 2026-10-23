import html
import random

import numpy as np
import streamlit as st
from scipy.optimize import linear_sum_assignment, linprog
import matplotlib.pyplot as plt


def format_linear_expression(coefficients, variables):
    terms = []
    for coefficient, variable in zip(coefficients, variables):
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        term = variable if magnitude == 1 else f"{magnitude:g}{variable}"
        if not terms:
            terms.append(term if coefficient > 0 else f"-{term}")
        else:
            terms.append(f" + {term}" if coefficient > 0 else f" - {term}")
    return "".join(terms) or "0"


def solve_linear_program(objective, coefficients, relations, limits, maximize):
    if (
        not objective
        or not coefficients
        or len(coefficients) != len(relations)
        or len(coefficients) != len(limits)
        or any(len(row) != len(objective) for row in coefficients)
        or any(relation not in ("<=", ">=", "=") for relation in relations)
    ):
        return None

    numeric_values = objective + [value for row in coefficients for value in row] + limits
    if not np.all(np.isfinite(numeric_values)):
        return None

    a_ub = []
    b_ub = []
    a_eq = []
    b_eq = []
    for row, relation, limit in zip(coefficients, relations, limits):
        if relation == "<=":
            a_ub.append(row)
            b_ub.append(limit)
        elif relation == ">=":
            a_ub.append([-value for value in row])
            b_ub.append(-limit)
        else:
            a_eq.append(row)
            b_eq.append(limit)

    objective_multiplier = -1 if maximize else 1
    return linprog(
        [value * objective_multiplier for value in objective],
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        A_eq=a_eq or None,
        b_eq=b_eq or None,
        bounds=[(0, None)] * len(objective),
        method="highs",
    )


def generate_practice_question(topic):
    if topic == "Aleatorio":
        topic = random.choice(["Programación lineal", "EOQ y punto de reorden"])

    if topic == "Programación lineal":
        maximize = random.choice([True, False])
        objective = [random.randint(2, 12), random.randint(2, 12)]
        coefficients = [
            [random.randint(1, 6), random.randint(1, 6)],
            [random.randint(1, 6), random.randint(1, 6)],
        ]
        limits = [random.randint(20, 60), random.randint(20, 60)]
        relation = "<=" if maximize else ">="
        relations = [relation, relation]
        result = solve_linear_program(
            objective,
            coefficients,
            relations,
            limits,
            maximize,
        )
        if result is None or not result.success:
            raise RuntimeError("No se pudo generar una práctica de programación lineal.")

        objective_name = "Z" if maximize else "C"
        objective_word = "Maximizar" if maximize else "Minimizar"
        relation_text = "≤" if maximize else "≥"
        constraints = ", ".join(
            f"{row[0]}x + {row[1]}y {relation_text} {limit}"
            for row, limit in zip(coefficients, limits)
        )
        value = -result.fun if maximize else result.fun
        return {
            "topic": "Programación lineal",
            "prompt": (
                f"{objective_word} {objective_name} = {objective[0]}x + "
                f"{objective[1]}y, sujeto a: {constraints}, x, y ≥ 0. "
                "Encuentra x, y y el valor óptimo."
            ),
            "answer": [float(result.x[0]), float(result.x[1]), float(value)],
            "solution": (
                f"La solución óptima es x = {result.x[0]:.2f}, "
                f"y = {result.x[1]:.2f}. Sustituyendo en la función objetivo: "
                f"{objective_name} = {objective[0]}({result.x[0]:.2f}) + "
                f"{objective[1]}({result.x[1]:.2f}) = {value:.2f}. "
                "El optimizador comparó las soluciones factibles respetando "
                "todas las restricciones y la no negatividad."
            ),
        }

    demand = random.randrange(1000, 10001, 500)
    order_cost = random.randrange(20, 101, 10)
    holding_cost = random.randrange(1, 11)
    lead_time = random.randint(1, 10)
    daily_demand = demand / 365
    eoq = float(np.sqrt((2 * demand * order_cost) / holding_cost))
    reorder_point = daily_demand * lead_time
    return {
        "topic": "EOQ y punto de reorden",
        "prompt": (
            f"La demanda anual es D = {demand} unidades, el costo por pedido "
            f"S = Q{order_cost}, el mantenimiento anual H = Q{holding_cost} "
            f"por unidad y el tiempo de entrega L = {lead_time} días "
            "(año de 365 días). Calcula EOQ y punto de reorden."
        ),
        "answer": [eoq, reorder_point],
        "solution": (
            f"EOQ = √(2DS/H) = √(2 × {demand} × {order_cost} / "
            f"{holding_cost}) = {eoq:.2f} unidades. "
            f"Demanda diaria = D/365 = {demand}/365 = {daily_demand:.2f}; "
            f"punto de reorden = demanda diaria × L = {daily_demand:.2f} × "
            f"{lead_time} = {reorder_point:.2f} unidades."
        ),
    }


def get_feasible_vertices(coefficients, limits):
    candidates = [(0.0, 0.0)]
    for (a_x, a_y), limit in zip(coefficients, limits):
        if a_x != 0:
            candidates.append((limit / a_x, 0.0))
        if a_y != 0:
            candidates.append((0.0, limit / a_y))

    if len(coefficients) >= 2:
        for first in range(len(coefficients)):
            for second in range(first + 1, len(coefficients)):
                a_x, a_y = coefficients[first]
                b_x, b_y = coefficients[second]
                determinant = a_x * b_y - a_y * b_x
                if abs(determinant) <= 1e-12:
                    continue
                first_limit = limits[first]
                second_limit = limits[second]
                candidates.append((
                    (first_limit * b_y - a_y * second_limit) / determinant,
                    (a_x * second_limit - first_limit * b_x) / determinant,
                ))

    feasible = []
    for x_value, y_value in candidates:
        scale = max(1.0, abs(x_value), abs(y_value))
        if x_value < -1e-8 * scale or y_value < -1e-8 * scale:
            continue
        if all(
            a_x * x_value + a_y * y_value
            <= limit + 1e-8 * max(1.0, abs(limit), abs(a_x * x_value), abs(a_y * y_value))
            for (a_x, a_y), limit in zip(coefficients, limits)
        ):
            point = (max(0.0, x_value), max(0.0, y_value))
            if not any(np.allclose(point, existing, rtol=1e-8, atol=1e-8) for existing in feasible):
                feasible.append(point)
    return feasible


def solve_simplex_max(objective, coefficients, limits, max_iterations=100):
    variable_count = len(objective)
    constraint_count = len(limits)
    if (
        not coefficients
        or len(coefficients) != constraint_count
        or any(len(row) != variable_count for row in coefficients)
        or any(limit < 0 for limit in limits)
    ):
        return "invalid", None, None, []

    tableau = np.zeros((constraint_count + 1, variable_count + constraint_count + 1))
    tableau[:constraint_count, :variable_count] = coefficients
    tableau[:constraint_count, variable_count:variable_count + constraint_count] = np.eye(constraint_count)
    tableau[:constraint_count, -1] = limits
    tableau[-1, :variable_count] = -np.asarray(objective, dtype=float)
    basis = list(range(variable_count, variable_count + constraint_count))
    steps = [{"tableau": tableau.copy(), "basis": basis.copy(), "pivot": None}]
    tolerance = 1e-9

    for _ in range(max_iterations):
        entering_candidates = np.flatnonzero(tableau[-1, :-1] < -tolerance)
        if len(entering_candidates) == 0:
            solution = np.zeros(variable_count)
            for row, basic_variable in enumerate(basis):
                if basic_variable < variable_count:
                    solution[basic_variable] = tableau[row, -1]
            return "optimal", solution, tableau[-1, -1], steps

        entering = int(entering_candidates[0])
        ratios = []
        for row in range(constraint_count):
            pivot_value = tableau[row, entering]
            if pivot_value > tolerance:
                ratios.append((tableau[row, -1] / pivot_value, basis[row], row))
        if not ratios:
            return "unbounded", None, None, steps

        minimum_ratio = min(ratio for ratio, _, _ in ratios)
        _, _, leaving_row = min(
            (ratio, basic_variable, row)
            for ratio, basic_variable, row in ratios
            if ratio <= minimum_ratio + tolerance
        )
        leaving_variable = basis[leaving_row]
        pivot_value = tableau[leaving_row, entering]
        tableau[leaving_row] /= pivot_value
        for row in range(constraint_count + 1):
            if row != leaving_row:
                tableau[row] -= tableau[row, entering] * tableau[leaving_row]
        basis[leaving_row] = entering
        steps.append({
            "tableau": tableau.copy(),
            "basis": basis.copy(),
            "pivot": (entering, leaving_variable, minimum_ratio),
        })

    return "iteration_limit", None, None, steps


# ----------------- CONFIGURACIÓN DE PÁGINA -----------------
st.set_page_config(
    page_title="MyGiAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- ESTILOS CSS PERSONALIZADOS (CORREGIDO PARA MODO OSCURO) -----------------
st.markdown("""
<style>
    /* Estilo principal de las cabeceras */
    .main-header {
        font-size: 2.3rem;
        color: var(--primary-color);
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .app-brand {
        font-size: clamp(4rem, 13vw, 8rem);
        line-height: 0.95;
        color: var(--primary-color);
        font-weight: 900;
        letter-spacing: 0.07em;
        text-shadow: 0 5px 18px rgba(0, 0, 0, 0.28), 0 0 30px color-mix(in srgb, var(--primary-color) 35%, transparent);
        margin: 0.4rem 0 1.2rem;
    }
    .app-brand::after {
        content: "";
        display: block;
        width: clamp(120px, 22vw, 220px);
        height: 8px;
        margin-top: 20px;
        border-radius: 999px;
        background-color: var(--primary-color);
        box-shadow: 0 3px 14px color-mix(in srgb, var(--primary-color) 55%, transparent);
    }
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-color);
        margin-bottom: 1.5rem;
    }
    
    /* Tarjetas personalizadas adaptadas a modo oscuro/claro */
    .custom-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.12);
        margin-bottom: 20px;
    }
    .info-card {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid var(--primary-color);
        color: var(--text-color);
        margin-bottom: 15px;
    }
    
    /* Personalización de métricas */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: var(--primary-color);
        font-weight: bold;
    }
    
    /* Botones primarios redondeados */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

</style>
""", unsafe_allow_html=True)

# ----------------- BARRA LATERAL (PANEL DE CONTROL) -----------------
st.sidebar.title("🎛️ Panel de Control")
st.sidebar.markdown("---")

# Información del Grupo en Expander Visual
with st.sidebar.expander("ℹ️ Información del Grupo", expanded=False):
    st.write("**Integrantes:**")
    st.text("• Mynor René Rivas Coronado")
    st.text("• Giancarlo Emmanuel Ortiz Marín")
    st.write("**Asignatura:** Investigación de Operaciones")
    st.write("**Sección:** A")
    st.write("**Universidad:** Mariano Gálvez de Guatemala")

st.sidebar.markdown("---")

modulo = st.sidebar.radio(
    "Selecciona un Módulo",
    [
        "🏠 Inicio",
        "📐 Modelación",
        "📈 Programación Lineal",
        "📊 Método Gráfico",
        "⚙️ Método Simplex",
        "🚚 Transporte y Asignación",
        "📦 Inventarios (EOQ)",
        "📝 Ejercicios de Práctica"
    ]
)

# ----------------- MÓDULO 1: INICIO (CHAT INTERACTIVO ESTILO WHATSAPP) -----------------
if modulo == "🏠 Inicio":
    import re
    
    st.markdown('<p class="app-brand">MyGiAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Universidad Mariano Gálvez de Guatemala | Facultad de Ingeniería de Sistemas</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="custom-card">
        <h3>💬 Centro de IA y Chat Académico</h3>
        <p>¡Hola! Soy el agente virtual de <b>MyGiAI</b>. Estoy programado por Giancarlo y Mynor para ayudarte a resolver dudas de Investigación de Operaciones, interpretar modelos lineales o guiarte en el uso de la plataforma. ¡Pregúntame lo que quieras!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Contenedor general del mensaje del usuario (Derecha) */
        .chat-row-user {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 12px;
            width: 100%;
        }
        /* Burbuja del usuario */
        .chat-bubble-user {
            background-color: #2e7d32; /* Verde WhatsApp / Operaciones */
            color: #ffffff !important;
            padding: 12px 16px;
            border-radius: 16px 16px 2px 16px;
            max-width: 75%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            font-size: 0.98rem;
            line-height: 1.4;
        }
        
        /* Contenedor general del mensaje del asistente (Izquierda) */
        .chat-row-assistant {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 12px;
            width: 100%;
        }
        /* Burbuja del asistente */
        .chat-bubble-assistant {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 12px 16px;
            border-radius: 16px 16px 16px 2px;
            max-width: 75%;
            box-shadow: 0 2px 5px rgba(0,0,0,0.15);
            border-left: 4px solid var(--primary-color);
            font-size: 0.98rem;
            line-height: 1.4;
        }
        
        /* Ocultar los avatares e íconos nativos de Streamlit para usar nuestras burbujas limpias */
        div[data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        div[data-testid="stChatMessageContent"] {
            padding: 0 !important;
        }
        div[data-testid="stChatMessageAvatar"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "¡Hola! Soy el asistente de MyGiAI. ¿En qué problema de optimización o inventarios te puedo asesorar hoy?"}
        ]

    for mensaje in st.session_state.chat_history:
        if mensaje["role"] == "user":
            contenido_usuario = html.escape(mensaje["content"])
            st.markdown(f"""
            <div class="chat-row-user">
                <div class="chat-bubble-user">{contenido_usuario}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            contenido_html = mensaje["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class="chat-row-assistant">
                <div class="chat-bubble-assistant">{contenido_html}</div>
            </div>
            """, unsafe_allow_html=True)

    if usuario_input := st.chat_input("Escribe tu pregunta o problema aquí..."):
        
        st.session_state.chat_history.append({"role": "user", "content": usuario_input})
        contenido_usuario = html.escape(usuario_input)
        st.markdown(f"""
        <div class="chat-row-user">
            <div class="chat-bubble-user">{contenido_usuario}</div>
        </div>
        """, unsafe_allow_html=True)
            
        with st.spinner("MyGiAI está procesando tu consulta..."):
            query = usuario_input.lower()
            respuesta_ia = ""
            
            match_x = re.search(r'(-?\d+)\s*x', query)
            match_y = re.search(r'(-?\d+)\s*y', query)
            restricciones = re.findall(r'(-?\d+)\s*x\s*[\+\-]\s*(-?\d+)\s*y\s*(<=|<=|=)\s*(-?\d+)', query)
            
            if match_x and match_y and len(restricciones) >= 2:
                c1 = float(match_x.group(1))
                c2 = float(match_y.group(1))
                tipo = "Minimizar" if "min" in query else "Maximizar"
                mult = -1 if tipo == "Maximizar" else 1
                
                r1_x, r1_y, _, r1_lim = restricciones[0]
                r2_x, r2_y, _, r2_lim = restricciones[1]
                
                res_t = linprog([c1 * mult, c2 * mult], A_ub=[[float(r1_x), float(r1_y)], [float(r2_x), float(r2_y)]], b_ub=[float(r1_lim), float(r2_lim)], bounds=[(0, None), (0, None)], method='highs')
                
                if res_t.success:
                    z_val = -res_t.fun if tipo == "Maximizar" else res_t.fun
                    respuesta_ia = f"📊 <b>¡Entendido! Detecté un modelo de {tipo} lineal.</b><br><br><b>Ecuación analizada:</b> {tipo} Z = {c1}x + {c2}y.<br><br><b>Resultados del cálculo:</b><br>• Variable X = <b>{res_t.x[0]:.2f}</b><br>• Variable Y = <b>{res_t.x[1]:.2f}</b><br>• Valor óptimo de la función objetivo Z = <b>Q{z_val:.2f}</b>.<br><br>Puedes validar la gráfica de este problema exacto y sus vértices en el menú izquierdo seleccionando el <b>Método Gráfico</b>."
                else:
                    respuesta_ia = "⚠️ Detecté la estructura de tu modelo de programación lineal, pero matemáticamente el sistema me indica que no cuenta con soluciones factibles en la región real."
            
            elif "hola" in query or "buenos dias" in query or "buenas tardes" in query:
                respuesta_ia = "¡Hola! Un gusto saludarte. Soy MyGiAI, el asistente virtual desarrollado para la asignatura de Investigación de Operaciones. ¿Qué tema deseas que revisemos hoy?"
            elif "simplex" in query or "método simplex" in query:
                respuesta_ia = "El <b>Método Simplex</b> es un procedimiento iterativo algebraico que permite mejorar la solución de la función objetivo en cada paso. Se mueve a través de los vértices de la región factible usando variables de holgura. Puedes usar nuestro calculador dedicado en la barra lateral izquierda."
            elif "eoq" in query or "inventario" in query or "reorden" in query:
                respuesta_ia = "El modelo <b>EOQ (Cantidad Económica de Pedido)</b> busca el equilibrio perfecto entre los costos de ordenar y los costos de mantener inventario en el almacén. Si deseas calcularlo con tus propios datos, haz clic en el módulo <b>Inventarios (EOQ)</b>."
            elif "transporte" in query or "asignación" in query:
                respuesta_ia = "Los modelos de <b>Transporte y Asignación</b> optimizan la distribución logística. El de transporte minimiza los fletes desde orígenes hacia destinos, y el de asignación empareja tareas con trabajadores buscando la máxima eficiencia."
            elif "quien te creo" in query or "creadores" in query or "integrantes" in query:
                respuesta_ia = "Fui desarrollado con orgullo por los ingenieros <b>Mynor René Rivas Coronado</b> y <b>Giancarlo Emmanuel Ortiz Marín</b>, estudiantes de la sección A de la Facultad de Ingeniería de la Universidad Mariano Gálvez de Guatemala."
            else:
                respuesta_ia = "💬 Recibí tu consulta, pero para darte una asesoría exacta, por favor mencióname alguna palabra clave relacionada (como <i>Simplex, EOQ, Transporte, Método Gráfico</i>) o escribe un modelo lineal con la estructura estándar (ejemplo: `Maximizar Z = 40x + 30y sujeto a: 2x+1y<=100, 1x+2y<=80`)."
            
            st.session_state.chat_history.append({"role": "assistant", "content": respuesta_ia})
            st.rerun()

# ----------------- MÓDULO 2: MODELACIÓN -----------------
elif modulo == "📐 Modelación":
    st.markdown('<p class="main-header">📐 Módulo Interactivo de Modelación</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Define la función objetivo y agrega hasta 10 restricciones para formular tu modelo.</p>', unsafe_allow_html=True)
    
    tab_calc, tab_info = st.tabs(["🚀 Formulación Interactiva", "📖 Marco Teórico"])
    
    with tab_calc:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            with st.container(border=True):
                st.subheader("Variables de Decisión")
                nombre_producto_1 = st.text_input("Nombre de Variable 1:", value="", placeholder="Ej. Escritorio Estándar (x)")
                utilidad_1 = st.number_input("Coeficiente de x en Z:", value=None, placeholder="Ej. 40")
                
                nombre_producto_2 = st.text_input("Nombre de Variable 2:", value="", placeholder="Ej. Escritorio Ejecutivo (y)")
                utilidad_2 = st.number_input("Coeficiente de y en Z:", value=None, placeholder="Ej. 30")
                
        with col_m2:
            with st.container(border=True):
                st.subheader("Configuración del Modelo")
                tipo_modelo = st.radio("Objetivo:", ["Maximizar", "Minimizar"], key="modeling_objective")
                cantidad_restricciones = st.number_input(
                    "Cantidad de restricciones:",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    key="modeling_constraint_count",
                )

        st.markdown("#### Restricciones")
        restricciones_modelo = []
        for indice in range(int(cantidad_restricciones)):
            st.markdown(f"**Restricción {indice + 1}**")
            col_rx, col_ry, col_signo, col_limite = st.columns([2, 2, 1, 2])
            with col_rx:
                coef_x = st.number_input(
                    f"Coeficiente de x (R{indice + 1}):",
                    value=None,
                    key=f"modeling_a_x_{indice}",
                )
            with col_ry:
                coef_y = st.number_input(
                    f"Coeficiente de y (R{indice + 1}):",
                    value=None,
                    key=f"modeling_a_y_{indice}",
                )
            with col_signo:
                relacion = st.selectbox(
                    f"Relación (R{indice + 1}):",
                    ["<=", ">=", "="],
                    key=f"modeling_relation_{indice}",
                )
            with col_limite:
                limite = st.number_input(
                    f"Límite (R{indice + 1}):",
                    value=None,
                    key=f"modeling_limit_{indice}",
                )
            restricciones_modelo.append((coef_x, coef_y, relacion, limite))

        if st.button("📝 Formular Modelo Matemático", use_container_width=True):
            if (
                not nombre_producto_1
                or not nombre_producto_2
                or utilidad_1 is None
                or utilidad_2 is None
                or any(None in restriccion for restriccion in restricciones_modelo)
            ):
                st.error("⚠️ Advertencia: Por favor completa todos los campos requeridos para generar el modelo.")
            else:
                st.success("¡Modelo formulado con éxito!")
                with st.container(border=True):
                    st.markdown("#### 📋 Estructura Matemática Resultante:")
                    st.write(f"**Variables:** $x$ = {nombre_producto_1}, $y$ = {nombre_producto_2}")
                    objetivo_tex = r"\max" if tipo_modelo == "Maximizar" else r"\min"
                    st.latex(
                        f"{objetivo_tex} Z = "
                        f"{format_linear_expression([utilidad_1, utilidad_2], ['x', 'y'])}"
                    )
                    st.markdown("**Sujeto a:**")
                    for coef_x, coef_y, relacion, limite in restricciones_modelo:
                        signo_tex = {"<=": r"\le", ">=": r"\ge", "=": "="}[relacion]
                        st.latex(
                            f"{format_linear_expression([coef_x, coef_y], ['x', 'y'])} "
                            f"{signo_tex} {limite:g}"
                        )
                    st.latex(r"x, y \ge 0 \quad (\text{Condición de no negatividad})")
                    
    with tab_info:
        st.markdown("""
        <div class="custom-card">
            <h4>💡 Componentes de un Modelo de Programación Lineal</h4>
            <p><strong>1. Variables de Decisión:</strong> Las incógnitas a determinar ($x, y$).</p>
            <p><strong>2. Función Objetivo:</strong> La meta matemática expresada como maximización (ganancias) o minimización (costos).</p>
            <p><strong>3. Restricciones:</strong> Limitaciones físicas, financieras o temporales del sistema.</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- MÓDULO 3: PROGRAMACIÓN LINEAL -----------------
elif modulo == "📈 Programación Lineal":
    st.markdown('<p class="main-header">📈 Programación Lineal</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Calcula la solución óptima ingresando coeficientes y límites de restricciones.</p>', unsafe_allow_html=True)
    
    col_input, col_config = st.columns([2, 1])
    
    with col_config:
        with st.container(border=True):
            st.subheader("⚙️ Configuración")
            tipo_opt = st.radio("Objetivo de Optimización:", ["Maximizar", "Minimizar"])
    
    with col_input:
        with st.container(border=True):
            st.subheader("🎯 Coeficientes de la Función Objetivo (Z)")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                c1 = st.number_input("Coeficiente de x:", value=None, placeholder="Ej. 40")
            with col_c2:
                c2 = st.number_input("Coeficiente de y:", value=None, placeholder="Ej. 30")

            cantidad_restricciones_pl = st.number_input(
                "Cantidad de restricciones:",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                key="linear_program_constraint_count",
            )

    restricciones_pl = []
    for indice in range(int(cantidad_restricciones_pl)):
        st.markdown(f"**Restricción {indice + 1}**")
        col_rx, col_ry, col_signo, col_limite = st.columns([2, 2, 1, 2])
        with col_rx:
            coef_x = st.number_input(
                f"Coef. x (R{indice + 1})",
                value=None,
                key=f"linear_program_a_x_{indice}",
            )
        with col_ry:
            coef_y = st.number_input(
                f"Coef. y (R{indice + 1})",
                value=None,
                key=f"linear_program_a_y_{indice}",
            )
        with col_signo:
            relacion = st.selectbox(
                f"Relación (R{indice + 1})",
                ["<=", ">=", "="],
                key=f"linear_program_relation_{indice}",
            )
        with col_limite:
            limite = st.number_input(
                f"Límite (R{indice + 1})",
                value=None,
                key=f"linear_program_limit_{indice}",
            )
        restricciones_pl.append((coef_x, coef_y, relacion, limite))

    if st.button("🚀 Calcular Solución Óptima", use_container_width=True):
        if c1 is None or c2 is None or any(
            None in restriccion for restriccion in restricciones_pl
        ):
            st.error("⚠️ Advertencia: Hay campos vacíos. Por favor ingresa todos los valores numéricos.")
        else:
            coefficients = [[row[0], row[1]] for row in restricciones_pl]
            relations = [row[2] for row in restricciones_pl]
            limits = [row[3] for row in restricciones_pl]
            res = solve_linear_program(
                [c1, c2],
                coefficients,
                relations,
                limits,
                tipo_opt == "Maximizar",
            )

            if res is None:
                st.error("No se pudo interpretar el modelo. Revisa las restricciones ingresadas.")
            elif res.success:
                val_z = -res.fun if tipo_opt == "Maximizar" else res.fun
                st.success("¡Solución óptima encontrada exitosamente!")
                with st.container(border=True):
                    col_res1, col_res2, col_res3 = st.columns(3)
                    col_res1.metric("Valor Óptimo (x)", f"{res.x[0]:.2f}")
                    col_res2.metric("Valor Óptimo (y)", f"{res.x[1]:.2f}")
                    col_res3.metric(f"Resultado {tipo_opt} (Z)", f"Q{val_z:.2f}")
                    st.markdown("#### Procedimiento y comprobación")
                    st.latex(
                        f"Z = {format_linear_expression([c1, c2], ['x', 'y'])}"
                    )
                    st.write(
                        f"Sustitución: Z = {c1:g}({res.x[0]:.2f}) "
                        f"+ {c2:g}({res.x[1]:.2f}) = {val_z:.2f}."
                    )
                    st.markdown("**Comprobación de las restricciones:**")
                    relation_symbols = {"<=": "≤", ">=": "≥", "=": "="}
                    for index, (row, relation, limit) in enumerate(
                        zip(coefficients, relations, limits),
                        start=1,
                    ):
                        activity = float(np.dot(row, res.x))
                        expression = format_linear_expression(row, ["x", "y"])
                        st.write(
                            f"R{index}: {expression} = "
                            f"{row[0]:g}({res.x[0]:.2f}) + "
                            f"{row[1]:g}({res.x[1]:.2f}) = {activity:.2f} "
                            f"{relation_symbols[relation]} {limit:g}."
                        )
                        if relation == "<=":
                            st.caption(f"Holgura (recurso no utilizado): {limit - activity:.2f}.")
                        elif relation == ">=":
                            st.caption(f"Excedente sobre el mínimo requerido: {activity - limit:.2f}.")
                        else:
                            st.caption("La igualdad se satisface dentro de la tolerancia numérica del solver.")
                    st.info(
                        "La solución se obtiene optimizando la función objetivo bajo las "
                        "restricciones y la condición x, y ≥ 0."
                    )
            elif res.status == 2:
                st.error("El modelo no tiene solución factible: las restricciones son incompatibles.")
            elif res.status == 3:
                st.error("El modelo es no acotado: la función objetivo puede mejorar sin límite.")
            else:
                st.error(f"No se pudo resolver el modelo: {res.message}")

# ----------------- MÓDULO 4: MÉTODO GRÁFICO -----------------
elif modulo == "📊 Método Gráfico":
    st.header("📊 Método Gráfico Interactivo (2 Variables)")
    st.markdown('<p class="sub-header">Grafica la región factible e identifica los vértices de solución óptima.</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            g_c1 = st.number_input("Función Objetivo - Coef. x:", value=None, placeholder="Ej. 40")
        with col_f2:
            g_c2 = st.number_input("Función Objetivo - Coef. y:", value=None, placeholder="Ej. 30")
        objetivo_grafico = st.radio(
            "Objetivo de optimización:",
            ["Maximizar", "Minimizar"],
            horizontal=True,
            key="graph_objective",
        )

        st.markdown(r"**Matriz de Restricciones ($\le$):**")
        col_mg1, col_mg2, col_mg3 = st.columns(3)
        with col_mg1:
            mg_a1 = st.number_input("R1: Coef. x", value=None, placeholder="Ej. 2", key="mg_a1")
        with col_mg2:
            mg_b1 = st.number_input("R1: Coef. y", value=None, placeholder="Ej. 1", key="mg_b1")
        with col_mg3:
            mg_c1 = st.number_input("R1: Límite", value=None, placeholder="Ej. 100", key="mg_c1")

        col_mg4, col_mg5, col_mg6 = st.columns(3)
        with col_mg4:
            mg_a2 = st.number_input("R2: Coef. x", value=None, placeholder="Ej. 1", key="mg_a2")
        with col_mg5:
            mg_b2 = st.number_input("R2: Coef. y", value=None, placeholder="Ej. 2", key="mg_b2")
        with col_mg6:
            mg_c2 = st.number_input("R2: Límite", value=None, placeholder="Ej. 80", key="mg_c2")

    if st.button("🖼️ Generar Gráfica Dinámica", use_container_width=True):
        if None in [g_c1, g_c2, mg_a1, mg_b1, mg_c1, mg_a2, mg_b2, mg_c2]:
            st.error("⚠️ Advertencia: Por favor llena todos los campos para poder generar la gráfica.")
        else:
            graph_coefficients = [[mg_a1, mg_b1], [mg_a2, mg_b2]]
            graph_limits = [mg_c1, mg_c2]
            objective = [g_c1, g_c2]
            objective_multiplier = -1 if objetivo_grafico == "Maximizar" else 1
            result = linprog(
                [coefficient * objective_multiplier for coefficient in objective],
                A_ub=graph_coefficients,
                b_ub=graph_limits,
                bounds=[(0, None), (0, None)],
                method="highs",
            )

            if result.status == 2:
                st.error("No existe una región factible con las restricciones ingresadas.")
            elif result.status not in (0, 3):
                st.error(f"No se pudo resolver el modelo: {result.message}")
            else:
                valid_vertices = get_feasible_vertices(graph_coefficients, graph_limits)
                if not valid_vertices:
                    st.error("No se encontraron vértices factibles para representar el modelo.")
                else:
                    if result.status == 3:
                        st.warning(
                            f"La región factible es no acotada en la dirección que se intenta "
                            f"{objetivo_grafico.lower()}; no existe un valor óptimo finito. "
                            "La gráfica muestra una ventana de la región."
                        )
                    else:
                        best_value = (
                            -result.fun if objetivo_grafico == "Maximizar" else result.fun
                        )
                        st.success(
                            f"Óptimo encontrado: x = {result.x[0]:.2f}, "
                            f"y = {result.x[1]:.2f}, Z = {best_value:.2f}."
                        )

                    x_intercepts = [
                        abs(limit / row[0])
                        for row, limit in zip(graph_coefficients, graph_limits)
                        if row[0] != 0
                    ]
                    y_intercepts = [
                        abs(limit / row[1])
                        for row, limit in zip(graph_coefficients, graph_limits)
                        if row[1] != 0
                    ]
                    max_x = 1.2 * max(
                        [point[0] for point in valid_vertices] + x_intercepts + [1.0]
                    )
                    max_y = 1.2 * max(
                        [point[1] for point in valid_vertices] + y_intercepts + [1.0]
                    )
                    x_vals = np.linspace(0, max_x, 250)
                    y_vals = np.linspace(0, max_y, 250)
                    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
                    feasible_mask = np.ones_like(grid_x, dtype=bool)
                    for (coef_x, coef_y), limit in zip(graph_coefficients, graph_limits):
                        feasible_mask &= coef_x * grid_x + coef_y * grid_y <= limit + 1e-9

                    col_plot, col_points = st.columns([3, 2])
                    with col_plot:
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        fig.patch.set_alpha(0)
                        ax.set_facecolor("none")
                        ax.pcolormesh(
                            grid_x,
                            grid_y,
                            feasible_mask.astype(float),
                            shading="auto",
                            cmap="Greens",
                            vmin=0,
                            vmax=1,
                            alpha=0.3,
                        )
                        for index, ((coef_x, coef_y), limit) in enumerate(
                            zip(graph_coefficients, graph_limits),
                            start=1,
                        ):
                            label = (
                                f"{coef_x:g}x + {coef_y:g}y <= {limit:g}"
                            )
                            if coef_y != 0:
                                ax.plot(
                                    x_vals,
                                    (limit - coef_x * x_vals) / coef_y,
                                    label=label,
                                )
                            elif coef_x != 0:
                                ax.axvline(limit / coef_x, label=label)
                        for vertex_x, vertex_y in valid_vertices:
                            ax.plot(vertex_x, vertex_y, "ro", markersize=4)
                        if result.status == 0:
                            ax.plot(result.x[0], result.x[1], "k*", markersize=12, label="Óptimo")
                        ax.set_xlim(0, max_x)
                        ax.set_ylim(0, max_y)
                        ax.set_xlabel("Variable X")
                        ax.set_ylabel("Variable Y")
                        ax.grid(True, linestyle="--", alpha=0.35)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)

                    with col_points:
                        with st.container(border=True):
                            st.markdown("#### 📐 Evaluación de Vértices")
                            for vertex_x, vertex_y in valid_vertices:
                                vertex_value = (
                                    g_c1 * vertex_x + g_c2 * vertex_y
                                )
                                st.write(
                                    f"• **({vertex_x:.2f}, {vertex_y:.2f}):** "
                                    f"Z = {vertex_value:.2f}"
                                )
                            if result.status == 0:
                                st.divider()
                                st.info(
                                    f"Objetivo: **{objetivo_grafico}** "
                                    f"(x = {result.x[0]:.2f}, y = {result.x[1]:.2f}, "
                                    f"Z = {best_value:.2f})."
                                )
                                st.write(
                                    "Sustitución en la función objetivo: "
                                    f"Z = {g_c1:g}({result.x[0]:.2f}) + "
                                    f"{g_c2:g}({result.x[1]:.2f}) = {best_value:.2f}."
                                )

# ----------------- MÓDULO 5: MÉTODO SIMPLEX -----------------
elif modulo == "⚙️ Método Simplex":
    st.markdown('<p class="main-header">⚙️ Método Simplex Paso a Paso</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Resuelve modelos de maximización con restricciones '
        '≤ y lado derecho no negativo, mostrando cada pivote del tableau.</p>',
        unsafe_allow_html=True,
    )
    
    with st.container(border=True):
        col_sx1, col_sx2 = st.columns(2)
        with col_sx1:
            sx_c1 = st.number_input("Coeficiente x en Z:", value=None, placeholder="Ej. 40", key="sx_c1")
        with col_sx2:
            sx_c2 = st.number_input("Coeficiente y en Z:", value=None, placeholder="Ej. 30", key="sx_c2")
            
        col_sa1, col_sa2, col_sa3 = st.columns(3)
        with col_sa1:
            sx_a1 = st.number_input("Coef. x (R1)", value=None, placeholder="Ej. 2", key="sx_a1")
        with col_sa2:
            sx_a2 = st.number_input("Coef. y (R1)", value=None, placeholder="Ej. 1", key="sx_a2")
        with col_sa3:
            sx_b1 = st.number_input("Límite (R1)", value=None, placeholder="Ej. 100", key="sx_b1")

        col_sa4, col_sa5, col_sa6 = st.columns(3)
        with col_sa4:
            sx_a3 = st.number_input("Coef. x (R2)", value=None, placeholder="Ej. 1", key="sx_a3")
        with col_sa5:
            sx_a4 = st.number_input("Coef. y (R2)", value=None, placeholder="Ej. 2", key="sx_a4")
        with col_sa6:
            sx_b2 = st.number_input("Límite (R2)", value=None, placeholder="Ej. 80", key="sx_b2")

    if st.button("🚀 Resolver y Explicar por Método Simplex", use_container_width=True):
        if None in [sx_c1, sx_c2, sx_a1, sx_a2, sx_b1, sx_a3, sx_a4, sx_b2]:
            st.error("⚠️ Advertencia: Faltan datos por ingresar en los campos numéricos.")
        elif sx_b1 < 0 or sx_b2 < 0:
            st.error(
                "El método Simplex de este módulo requiere límites no negativos "
                "para iniciar con variables de holgura. Convierte las restricciones "
                "antes de ingresar los datos."
            )
        else:
            simplex_status, simplex_solution, z_val, simplex_steps = solve_simplex_max(
                [sx_c1, sx_c2],
                [[sx_a1, sx_a2], [sx_a3, sx_a4]],
                [sx_b1, sx_b2],
            )

            if simplex_status == "optimal":
                x_val, y_val = simplex_solution
                holguras = [
                    limit - (row[0] * x_val + row[1] * y_val)
                    for row, limit in zip(
                        [[sx_a1, sx_a2], [sx_a3, sx_a4]],
                        [sx_b1, sx_b2],
                    )
                ]
                tab_s1, tab_s2, tab_s3 = st.tabs(["📝 1. Forma Estándar", "🔄 2. Procedimiento Simplex", "🎯 3. Resultados y Holguras"])
                
                with tab_s1:
                    st.write(
                        "Se agregan variables de holgura ($s_1, s_2$) para convertir "
                        "las restricciones ≤ en ecuaciones. La base inicial es factible "
                        "porque los límites ingresados son no negativos."
                    )
                    st.latex(
                        "Z - ("
                        f"{format_linear_expression([sx_c1, sx_c2], ['x', 'y'])}"
                        ") + 0s_1 + 0s_2 = 0"
                    )
                    st.latex(
                        f"{format_linear_expression([sx_a1, sx_a2], ['x', 'y'])} "
                        f"+ s_1 = {sx_b1}"
                    )
                    st.latex(
                        f"{format_linear_expression([sx_a3, sx_a4], ['x', 'y'])} "
                        f"+ s_2 = {sx_b2}"
                    )
                    
                with tab_s2:
                    st.markdown("""
                    Se usa la regla de Bland para elegir la primera variable con costo reducido
                    negativo y el cociente mínimo para elegir la variable saliente.
                    """)
                    tableau_columns = ["x", "y", "s1", "s2", "LD"]
                    for iteration, step in enumerate(simplex_steps):
                        if step["pivot"] is None:
                            st.markdown(f"**Tableau inicial (iteración {iteration})**")
                        else:
                            entering, leaving, ratio = step["pivot"]
                            variable_names = ["x", "y", "s1", "s2"]
                            st.markdown(
                                f"**Iteración {iteration}:** entra "
                                f"${variable_names[entering]}$, sale "
                                f"${variable_names[leaving]}$ "
                                f"(cociente mínimo = {ratio:.3f})."
                            )
                            st.caption(
                                "Se divide la fila pivote entre su elemento pivote; "
                                "después se elimina la variable entrante de las demás "
                                "filas para obtener el siguiente tableau."
                            )
                        tableau_rows = []
                        for row_index, row in enumerate(step["tableau"]):
                            if row_index == len(step["basis"]):
                                base_name = "Z"
                            else:
                                base_name = ["x", "y", "s1", "s2"][step["basis"][row_index]]
                            tableau_rows.append({
                                "Base": base_name,
                                **{
                                    column: f"{value:.3f}"
                                    for column, value in zip(tableau_columns, row)
                                },
                            })
                        st.dataframe(tableau_rows, hide_index=True, width="stretch")
                    
                with tab_s3:
                    col_v1, col_v2, col_v3 = st.columns(3)
                    col_v1.metric("Valor x", f"{x_val:.2f}")
                    col_v2.metric("Valor y", f"{y_val:.2f}")
                    col_v3.metric("Valor Máximo Z", f"Q{z_val:.2f}")
                    st.write(
                        f"Sustitución: Z = {sx_c1:g}({x_val:.2f}) + "
                        f"{sx_c2:g}({y_val:.2f}) = {z_val:.2f}."
                    )
                    
                    st.info(f"""
                    **Análisis de Recursos (Holguras):**
                    * **Holgura en Recurso 1 ($s_1$):** {max(0, holguras[0]):.2f} unidades no utilizadas.
                    * **Holgura en Recurso 2 ($s_2$):** {max(0, holguras[1]):.2f} unidades no utilizadas.
                    """)
            elif simplex_status == "unbounded":
                st.error("El modelo es no acotado: la función objetivo puede crecer sin límite.")
            elif simplex_status == "iteration_limit":
                st.error("El método alcanzó su límite de iteraciones sin encontrar el óptimo.")
            else:
                st.error("No se pudo construir una base inicial válida con los datos ingresados.")

# ----------------- MÓDULO 6: TRANSPORTE Y ASIGNACIÓN -----------------
elif modulo == "🚚 Transporte y Asignación":
    st.markdown('<p class="main-header">🚚 Transporte y Asignación de Recursos</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimización de redes logísticas de distribución y asignación de recursos.</p>', unsafe_allow_html=True)
    
    sub_modulo = st.segmented_control(
        "Elige la variante:",
        ["📦 Modelo de Transporte (Distribución)", "👥 Modelo de Asignación (Algoritmo Húngaro)"],
        default="📦 Modelo de Transporte (Distribución)"
    )
    
    st.markdown("---")
    
    if sub_modulo == "📦 Modelo de Transporte (Distribución)":
        with st.container(border=True):
            st.subheader("📦 Configuración de Oferta, Demanda y Costos")
            col_of1, col_of2 = st.columns(2)
            with col_of1:
                oferta_1 = st.number_input("Oferta Origen 1:", value=None, placeholder="Ej. 60")
                demanda_1 = st.number_input("Demanda Destino 1:", value=None, placeholder="Ej. 50")
            with col_of2:
                oferta_2 = st.number_input("Oferta Origen 2:", value=None, placeholder="Ej. 80")
                demanda_2 = st.number_input("Demanda Destino 2:", value=None, placeholder="Ej. 90")
                
            st.markdown("**Matriz de Costos Unitarios de Envío:**")
            col_cs1, col_cs2 = st.columns(2)
            with col_cs1:
                costo_11 = st.number_input("Costo O1 -> D1", value=None, placeholder="Ej. 2")
                costo_21 = st.number_input("Costo O2 -> D1", value=None, placeholder="Ej. 6")
            with col_cs2:
                costo_12 = st.number_input("Costo O1 -> D2", value=None, placeholder="Ej. 3")
                costo_22 = st.number_input("Costo O2 -> D2", value=None, placeholder="Ej. 1")
            
        if st.button("🧮 Calcular Plan de Transporte", use_container_width=True):
            if None in [oferta_1, oferta_2, demanda_1, demanda_2, costo_11, costo_12, costo_21, costo_22]:
                st.error("⚠️ Por favor llena todos los campos de transporte para calcular la distribución.")
            elif min(oferta_1, oferta_2, demanda_1, demanda_2) < 0:
                st.error("La oferta y la demanda deben ser valores no negativos.")
            elif min(costo_11, costo_12, costo_21, costo_22) < 0:
                st.error("Los costos unitarios no pueden ser negativos.")
            elif not np.isclose(
                oferta_1 + oferta_2,
                demanda_1 + demanda_2,
                rtol=1e-9,
                atol=1e-9,
            ):
                st.error(f"⚠️ Modelo desbalanceado (Oferta Total: {oferta_1+oferta_2} != Demanda Total: {demanda_1+demanda_2}).")
            else:
                c_trans = [costo_11, costo_12, costo_21, costo_22]
                A_eq = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
                b_eq = [oferta_1, oferta_2, demanda_1, demanda_2]
                
                res_t = linprog(c_trans, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
                
                if res_t.success:
                    st.success("¡Plan logístico optimizado!")
                    with st.container(border=True):
                        st.markdown("#### 📦 Distribución Óptima Recomendada:")
                        st.write(f"• Enviar **{res_t.x[0]:.1f}** unidades de Origen 1 a Destino 1.")
                        st.write(f"• Enviar **{res_t.x[1]:.1f}** unidades de Origen 1 a Destino 2.")
                        st.write(f"• Enviar **{res_t.x[2]:.1f}** unidades de Origen 2 a Destino 1.")
                        st.write(f"• Enviar **{res_t.x[3]:.1f}** unidades de Origen 2 a Destino 2.")
                        st.metric("Costo Mínimo Total de Transporte", f"Q{res_t.fun:.2f}")
                        route_names = ["O1 → D1", "O1 → D2", "O2 → D1", "O2 → D2"]
                        st.markdown("**Cálculo del costo:**")
                        cost_terms = []
                        for route, quantity, unit_cost in zip(
                            route_names,
                            res_t.x,
                            c_trans,
                        ):
                            st.write(
                                f"{route}: {quantity:.1f} unidades × "
                                f"Q{unit_cost:.2f} = Q{quantity * unit_cost:.2f}"
                            )
                            cost_terms.append(f"{quantity:.1f} × {unit_cost:.2f}")
                        st.info(
                            f"Costo total = {' + '.join(cost_terms)} = "
                            f"Q{res_t.fun:.2f}. La solución cumple la oferta de "
                            "cada origen y la demanda de cada destino."
                        )
                else:
                    st.error(f"No se pudo resolver el modelo de transporte: {res_t.message}")

    else:
        with st.container(border=True):
            st.subheader("👥 Matriz de Asignación 3x3 (Costos o Tiempos)")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                c_t1_a = st.number_input("T1 - Tarea A", value=None, placeholder="Ej. 9", key="c_t1_a")
                c_t2_a = st.number_input("T2 - Tarea A", value=None, placeholder="Ej. 6", key="c_t2_a")
                c_t3_a = st.number_input("T3 - Tarea A", value=None, placeholder="Ej. 5", key="c_t3_a")
            with col_m2:
                c_t1_b = st.number_input("T1 - Tarea B", value=None, placeholder="Ej. 2", key="c_t1_b")
                c_t2_b = st.number_input("T2 - Tarea B", value=None, placeholder="Ej. 4", key="c_t2_b")
                c_t3_b = st.number_input("T3 - Tarea B", value=None, placeholder="Ej. 8", key="c_t3_b")
            with col_m3:
                c_t1_c = st.number_input("T1 - Tarea C", value=None, placeholder="Ej. 7", key="c_t1_c")
                c_t2_c = st.number_input("T2 - Tarea C", value=None, placeholder="Ej. 3", key="c_t2_c")
                c_t3_c = st.number_input("T3 - Tarea C", value=None, placeholder="Ej. 6", key="c_t3_c")
            
        if st.button("🎯 Resolver Asignación Óptima", use_container_width=True):
            if None in [c_t1_a, c_t2_a, c_t3_a, c_t1_b, c_t2_b, c_t3_b, c_t1_c, c_t2_c, c_t3_c]:
                st.error("⚠️ Por favor llena todos los campos de la matriz para resolver la asignación.")
            else:
                cost_matrix = np.array([
                    [c_t1_a, c_t1_b, c_t1_c],
                    [c_t2_a, c_t2_b, c_t2_c],
                    [c_t3_a, c_t3_b, c_t3_c]
                ])
                if np.any(cost_matrix < 0):
                    st.error("Los costos o tiempos de asignación no pueden ser negativos.")
                else:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    total_assignment_cost = cost_matrix[row_ind, col_ind].sum()

                    st.success("¡Asignación óptima completada!")
                    with st.container(border=True):
                        tareas = ["Tarea A", "Tarea B", "Tarea C"]
                        assignment_terms = []
                        for r, c in zip(row_ind, col_ind):
                            value = cost_matrix[r, c]
                            st.write(
                                f"• Trabajador {r + 1} → {tareas[c]}: "
                                f"{value:.2f}"
                            )
                            assignment_terms.append(f"{value:.2f}")
                        st.metric("Costo / Tiempo Mínimo Total", f"{total_assignment_cost:.2f}")
                        st.info(
                            f"Objetivo: minimizar la suma de las asignaciones elegidas: "
                            f"{' + '.join(assignment_terms)} = {total_assignment_cost:.2f}. "
                            "Cada trabajador queda asignado a una sola tarea y cada tarea "
                            "recibe un solo trabajador."
                        )

# ----------------- MÓDULO 7: INVENTARIOS (EOQ) -----------------
elif modulo == "📦 Inventarios (EOQ)":
    st.markdown('<p class="main-header">📦 Modelos de Inventario Determinísticos (EOQ)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Calcula la Cantidad Económica de Pedido y el Punto de Reorden (ROP).</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        col_d, col_s, col_h = st.columns(3)
        with col_d:
            D = st.number_input("Demanda Anual (D):", value=None, placeholder="Ej. 1200")
        with col_s:
            S = st.number_input("Costo por Pedido (S):", value=None, placeholder="Ej. 50")
        with col_h:
            H = st.number_input("Costo Mantenimiento (H):", value=None, placeholder="Ej. 2")
            
        L = st.number_input("Tiempo de Entrega en Días (L):", value=None, placeholder="Ej. 5")
    
    if st.button("📦 Calcular Métricas EOQ", use_container_width=True):
        if None in [D, S, H, L] or D <= 0 or S <= 0 or H <= 0 or L <= 0:
            st.error("⚠️ Todos los campos deben estar completos y ser mayores que cero.")
        else:
            eoq = np.sqrt((2 * D * S) / H)
            d_diaria = D / 365.0
            rop = d_diaria * L
            num_pedidos = D / eoq
            
            st.success("¡Cálculos de inventario completados!")
            with st.container(border=True):
                col_e1, col_e2, col_e3 = st.columns(3)
                col_e1.metric("Lote Económico (EOQ)", f"{eoq:.2f} u")
                col_e2.metric("Punto de Reorden (ROP)", f"{rop:.2f} u")
                col_e3.metric("Pedidos / Año", f"{num_pedidos:.2f}")
                st.markdown("#### Sustitución en las fórmulas")
                st.latex(r"EOQ = \sqrt{\frac{2DS}{H}}")
                st.write(
                    f"EOQ = √((2 × {D:g} × {S:g}) / {H:g}) "
                    f"= **{eoq:.2f} unidades por pedido**."
                )
                st.latex(r"ROP = d \times L,\quad d = \frac{D}{365}")
                st.write(
                    f"Demanda diaria d = {D:g} / 365 = {d_diaria:.2f} unidades/día. "
                    f"ROP = {d_diaria:.2f} × {L:g} = **{rop:.2f} unidades**. "
                    f"Pedidos anuales = D / EOQ = {D:g} / {eoq:.2f} = "
                    f"{num_pedidos:.2f}."
                )
                st.caption(
                    "El cálculo del punto de reorden supone demanda uniforme y "
                    "un año de 365 días; no incluye inventario de seguridad."
                )

# ----------------- MÓDULO 8: EJERCICIOS DE PRÁCTICA -----------------
elif modulo == "📝 Ejercicios de Práctica":
    st.markdown('<p class="main-header">📝 Ejercicios, Prácticas y Autoevaluación</p>', unsafe_allow_html=True)

    busqueda = st.text_input(
        "🔍 Buscar ejercicio por palabra clave o tema:",
        value="",
        placeholder="Ej. EOQ, Maximización, fletes, granja...",
    ).strip()
    
    ejercicios = {
        "Ejercicio 1: Maximización de Escritorios": {
            "enunciado": "Maximizar Z = 40x + 60y sujeto a: 2x + y <= 100, x + 3y <= 120, x,y >= 0.",
            "solucion": "x = 36 escritorios estándar, y = 28 escritorios ejecutivos. Z máx = Q3,120.00."
        },
        "Ejercicio 2: Minimización de Mezcla de Grano": {
            "enunciado": "Minimizar C = 5x + 4y sujeto a: 3x + y >= 12, 2x + 4y >= 16, x,y >= 0.",
            "solucion": "x = 3.2 kg de Grano A, y = 2.4 kg de Grano B. Costo Mínimo = Q25.60."
        },
        "Ejercicio 3: EOQ de Cerraduras": {
            "enunciado": "D = 10,000 unidades, S = Q50, H = Q2. Calcule EOQ.",
            "solucion": "EOQ = sqrt((2 * 10000 * 50) / 2) = 707.11 unidades por pedido."
        },
        "Ejercicio 4: Punto de Reorden (ROP)": {
            "enunciado": "D = 10,000, L = 4 días, 365 días laborables. Calcule ROP.",
            "solucion": "d = 10,000 / 365 = 27.40 u/día. ROP = 27.40 * 4 = 110 unidades."
        },
        "Ejercicio 5: Transporte de Fábricas a Tiendas": {
            "enunciado": "Plantas O1 (Oferta 60) y O2 (Oferta 80). Destinos D1 (Demanda 50) y D2 (Demanda 90). Costos: O1->D1: 2, O1->D2: 3, O2->D1: 6, O2->D2: 1.",
            "solucion": "50 u de O1 a D1, 10 u de O1 a D2 y 80 u de O2 a D2. Costo Total = Q210.00."
        },
        "Ejercicio 6: Producción de Sillas y Mesas": {
            "enunciado": "Maximizar Z = 20x + 30y sujeto a: x + y <= 40, 2x + y <= 60, x,y >= 0.",
            "solucion": "x = 0, y = 40. Z máx = Q1,200.00."
        },
        "Ejercicio 7: Minimización de Costos en Granja": {
            "enunciado": "Minimizar C = 2x + 3y sujeto a: x + 2y >= 10, 2x + y >= 12, x,y >= 0.",
            "solucion": "x = 4.67, y = 2.67. Costo Mínimo = Q17.33."
        },
        "Ejercicio 8: Inventario EOQ para Calzado": {
            "enunciado": "Demanda D = 5,000, S = Q100, H = Q5. Calcule EOQ.",
            "solucion": "EOQ = sqrt((2 * 5000 * 100) / 5) = 447.21 unidades."
        },
        "Ejercicio 9: Asignación de Tareas": {
            "enunciado": "Minimizar tiempos de respuesta asignando 2 trabajadores a 2 máquinas. Tiempos (horas): Trabajador 1: Máquina A = 8, Máquina B = 4; Trabajador 2: Máquina A = 3, Máquina B = 9.",
            "matriz": [[8, 4], [3, 9]],
            "solucion": "Trabajador 1 a Máquina B (4 hrs), Trabajador 2 a Máquina A (3 hrs). Tiempo Total = 7 hrs."
        },
        "Ejercicio 10: Planificación de Horas Hombre": {
            "enunciado": "Maximizar Z = 50x + 40y sujeto a: 3x + 2y <= 120, x + y <= 50, x,y >= 0.",
            "solucion": "x = 20, y = 30. Z máx = Q2,200.00."
        }
    }

    ejercicios_filtrados = {
        titulo: ejercicio
        for titulo, ejercicio in ejercicios.items()
        if busqueda.casefold() in (
            titulo + " " + ejercicio["enunciado"]
        ).casefold()
    }
    if not ejercicios_filtrados:
        st.warning("No hay ejercicios que coincidan con la búsqueda. Prueba con otra palabra clave.")
    else:
        opcion_ej = st.selectbox(
            "Selecciona un ejercicio del catálogo:",
            list(ejercicios_filtrados),
        )

        with st.container(border=True):
            st.markdown("#### Enunciado:")
            st.write(ejercicios_filtrados[opcion_ej]["enunciado"])

            with st.expander("🔍 Ver Procedimiento y Respuesta Óptima"):
                st.success(ejercicios_filtrados[opcion_ej]["solucion"])
                if "matriz" in ejercicios_filtrados[opcion_ej]:
                    st.markdown("**Matriz de tiempos (horas):**")
                    assignment_matrix = np.array(ejercicios_filtrados[opcion_ej]["matriz"])
                    st.table({
                        "Trabajador": ["1", "2"],
                        "Máquina A": assignment_matrix[:, 0],
                        "Máquina B": assignment_matrix[:, 1],
                    })
                    assigned_workers, assigned_machines = linear_sum_assignment(assignment_matrix)
                    st.write("**Comprobación óptima:**")
                    for worker, machine in zip(assigned_workers, assigned_machines):
                        machine_name = "A" if machine == 0 else "B"
                        st.write(
                            f"• Trabajador {worker + 1} → Máquina {machine_name}: "
                            f"{assignment_matrix[worker, machine]} horas"
                        )
                    st.metric(
                        "Tiempo mínimo total",
                        f"{assignment_matrix[assigned_workers, assigned_machines].sum():.0f} horas",
                    )

    st.markdown("---")
    st.subheader("🧩 Generador de prácticas nuevas")
    st.write(
        "Genera un problema con datos aleatorios. Resuélvelo antes de consultar "
        "la solución y recibe retroalimentación inmediata."
    )
    practice_topic = st.selectbox(
        "Tema de la práctica:",
        ["Aleatorio", "Programación lineal", "EOQ y punto de reorden"],
    )
    st.caption(
        "Se admite un margen de redondeo de hasta 1% o 0.05 unidades, "
        "lo que sea mayor."
    )
    if st.button("🎲 Generar nueva práctica", use_container_width=True):
        st.session_state.practice_question = generate_practice_question(practice_topic)
        st.session_state.pop("practice_answer_x", None)
        st.session_state.pop("practice_answer_y", None)
        st.session_state.pop("practice_answer_z", None)
        st.session_state.pop("practice_answer_eoq", None)
        st.session_state.pop("practice_answer_rop", None)
        st.session_state.pop("practice_feedback", None)

    practice_question = st.session_state.get("practice_question")
    if practice_question:
        with st.container(border=True):
            st.markdown(f"**{practice_question['topic']}**")
            st.write(practice_question["prompt"])
            if practice_question["topic"] == "Programación lineal":
                answer_columns = st.columns(3)
                answer_x = answer_columns[0].number_input(
                    "Tu respuesta para x:",
                    value=None,
                    key="practice_answer_x",
                )
                answer_y = answer_columns[1].number_input(
                    "Tu respuesta para y:",
                    value=None,
                    key="practice_answer_y",
                )
                answer_z = answer_columns[2].number_input(
                    "Valor óptimo:",
                    value=None,
                    key="practice_answer_z",
                )
                user_answers = [answer_x, answer_y, answer_z]
                answer_labels = ["x", "y", "valor óptimo"]
            else:
                answer_columns = st.columns(2)
                answer_eoq = answer_columns[0].number_input(
                    "Tu respuesta para EOQ:",
                    value=None,
                    key="practice_answer_eoq",
                )
                answer_rop = answer_columns[1].number_input(
                    "Tu respuesta para ROP:",
                    value=None,
                    key="practice_answer_rop",
                )
                user_answers = [answer_eoq, answer_rop]
                answer_labels = ["EOQ", "ROP"]

            if st.button("✅ Revisar mis respuestas", use_container_width=True):
                if any(answer is None for answer in user_answers):
                    st.error("Completa todos los campos de respuesta antes de revisarlos.")
                else:
                    correct = all(
                        abs(actual - expected) <= max(0.05, abs(expected) * 0.01)
                        for actual, expected in zip(
                            user_answers,
                            practice_question["answer"],
                        )
                    )
                    if correct:
                        st.session_state.practice_feedback = (
                            "¡Excelente! Tus respuestas son correctas dentro de "
                            "la tolerancia de redondeo indicada."
                        )
                    else:
                        incorrect_labels = [
                            label
                            for label, actual, expected in zip(
                                answer_labels,
                                user_answers,
                                practice_question["answer"],
                            )
                            if abs(actual - expected) > max(0.05, abs(expected) * 0.01)
                        ]
                        st.session_state.practice_feedback = (
                            "Hay respuestas por corregir: "
                            + ", ".join(incorrect_labels)
                            + ". Revisa la solución paso a paso."
                        )

            if st.session_state.get("practice_feedback"):
                if st.session_state.practice_feedback.startswith("¡Excelente"):
                    st.success(st.session_state.practice_feedback)
                else:
                    st.warning(st.session_state.practice_feedback)
                with st.expander("📖 Consultar solución explicada"):
                    st.write(practice_question["solution"])
        
    st.markdown("---")
    st.subheader("🎲 Pregunta Interactiva de Autoevaluación")
    with st.container(border=True):
        st.write(r"¿Qué representa la variable de holgura ($s_i$) en una restricción de tipo menor o igual ($\le$)?")
        
        resp = st.radio("Selecciona tu opción:", [
            "El recurso no utilizado o sobrante",
            "El costo marginal del producto",
            "El beneficio máximo alcanzable",
            "El exceso sobre el requerimiento mínimo"
        ])
        
        if st.button("Verificar Respuesta"):
            if resp == "El recurso no utilizado o sobrante":
                st.success("🎉 ¡Correcto! La variable de holgura representa la cantidad de recurso disponible que no fue consumida.")
            else:
                st.error("❌ Incorrecto. Recuerda que en restricciones <=, la holgura mide la capacidad sobrante.")
