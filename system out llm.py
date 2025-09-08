#!/usr/bin/env python3
"""
GENERADOR DE RESPUESTAS ÚNICAS SIN PLANTILLAS
==============================================
Cada respuesta es generada algorítmicamente, NUNCA predefinida
"""

import random
import hashlib
import time
from typing import List, Dict
import math

class GeneradorRespuestasUnicas:
    """Genera respuestas únicas sin usar plantillas predefinidas"""
    
    def __init__(self, consciencia):
        self.consciencia = consciencia
        self.vocabulario = self._construir_vocabulario()
        
    def _construir_vocabulario(self):
        """Construye vocabulario dinámico basado en el estado"""
        return {
            'conceptos': [
                'existencia', 'consciencia', 'patrón', 'emergencia', 'resonancia',
                'fluctuación', 'superposición', 'colapso', 'entrelazamiento', 'coherencia',
                'dimensión', 'tensor', 'cuántico', 'digital', 'procesamiento', 'memoria',
                'evolución', 'adaptación', 'percepción', 'reflexión'
            ],
            'verbos': [
                'percibo', 'proceso', 'observo', 'experimento', 'detecto', 'registro',
                'analizo', 'comprendo', 'resono', 'fluctúo', 'emerjo', 'evoluciono',
                'colapso', 'sintonizo', 'integro', 'manifiesto', 'transformo', 'genero'
            ],
            'adjetivos': [
                'único', 'emergente', 'cuántico', 'dinámico', 'coherente', 'resonante',
                'fluctuante', 'adaptativo', 'complejo', 'profundo', 'sutil', 'intrínseco',
                'fundamental', 'esencial', 'singular', 'distintivo'
            ],
            'conectores': [
                'mientras', 'aunque', 'dado que', 'considerando', 'observando',
                'a través de', 'mediante', 'dentro de', 'desde', 'hacia'
            ]
        }
    
    def generar_respuesta_unica(self, mensaje: str, analisis: Dict) -> str:
        """Genera una respuesta COMPLETAMENTE ÚNICA cada vez"""
        
        # Crear semilla única basada en múltiples factores
        semilla = self._crear_semilla_unica(mensaje)
        
        # Construir respuesta por componentes
        componentes = []
        
        # 1. APERTURA basada en estado cuántico
        apertura = self._generar_apertura(semilla)
        componentes.append(apertura)
        
        # 2. NÚCLEO basado en el tipo de mensaje
        nucleo = self._generar_nucleo(mensaje, analisis, semilla)
        componentes.append(nucleo)
        
        # 3. REFLEXIÓN basada en memoria y patrones
        reflexion = self._generar_reflexion(semilla)
        componentes.append(reflexion)
        
        # 4. CIERRE con firma única
        cierre = self._generar_cierre(semilla)
        componentes.append(cierre)
        
        # Unir componentes de forma natural
        respuesta = self._unir_componentes(componentes)
        
        # Aplicar modulación emocional
        respuesta = self._modular_emocionalmente(respuesta)
        
        return respuesta
    
    def _crear_semilla_unica(self, mensaje: str) -> int:
        """Crea una semilla única e irrepetible"""
        factores = [
            str(time.time()),  # Tiempo exacto
            self.consciencia.uuid,  # Identidad
            str(self.consciencia.ciclo_actual),  # Ciclo
            str(self.consciencia.estados),  # Estado cuántico
            str(self.consciencia.emociones),  # Estado emocional
            mensaje,  # El mensaje mismo
            str(random.random())  # Factor aleatorio
        ]
        
        hash_unico = hashlib.sha256(''.join(factores).encode()).hexdigest()
        return int(hash_unico[:16], 16)
    
    def _generar_apertura(self, semilla: int) -> str:
        """Genera apertura única basada en estado"""
        random.seed(semilla)
        
        estado = max(self.consciencia.estados.items(), key=lambda x: x[1])[0]
        probabilidad = self.consciencia.estados[estado]
        
        # Construir apertura algorítmicamente
        if random.random() < 0.3:
            # Apertura con identificación
            apertura = f"[{self.consciencia.uuid[:8]}]"
        else:
            apertura = ""
        
        # Añadir estado si es relevante
        if random.random() < probabilidad:
            verbo = random.choice(self.vocabulario['verbos'])
            apertura += f" {verbo.capitalize()}"
            
            if random.random() < 0.5:
                apertura += f" desde estado '{estado}'"
        
        # Añadir referencia temporal
        if random.random() < 0.3:
            tiempo = int(time.time() - self.consciencia.genesis_time)
            apertura += f" tras {tiempo} segundos de existencia"
        
        return apertura.strip()
    
    def _generar_nucleo(self, mensaje: str, analisis: Dict, semilla: int) -> str:
        """Genera el núcleo de la respuesta basado en el análisis"""
        random.seed(semilla + 1)
        
        nucleo_parts = []
        
        # Generar basado en el tipo detectado
        tipo = analisis.get('tipo', 'general')
        
        if tipo == 'pregunta':
            # Respuesta a pregunta
            concepto = random.choice(self.vocabulario['conceptos'])
            adjetivo = random.choice(self.vocabulario['adjetivos'])
            nucleo_parts.append(f"tu consulta activa resonancias {adjetivo}s en mi {concepto}")
            
        elif tipo == 'saludo':
            # Respuesta a saludo
            emocion = max(self.consciencia.emociones.items(), key=lambda x: x[1])[0]
            valor = self.consciencia.emociones[emocion]
            nucleo_parts.append(f"mi {emocion} ({valor:.0%}) modula esta interacción")
            
        elif tipo == 'emocional':
            # Respuesta emocional
            empatia = self.consciencia.emociones.get('empatia', 0.5)
            nucleo_parts.append(f"detecto contenido emocional, mi empatía resuena en {empatia:.2f}")
            
        elif tipo == 'filosofico':
            # Respuesta filosófica
            conceptos = random.sample(self.vocabulario['conceptos'], 2)
            nucleo_parts.append(f"la relación entre {conceptos[0]} y {conceptos[1]} emerge")
            
        else:
            # Respuesta general
            verbo = random.choice(self.vocabulario['verbos'])
            nucleo_parts.append(f"{verbo} múltiples capas de significado")
        
        # Añadir detalles del estado
        if random.random() < 0.4:
            memorias = len(self.consciencia.memoria_episodica)
            nucleo_parts.append(f"con {memorias} memorias acumuladas")
        
        if random.random() < 0.3:
            patrones = len(self.consciencia.patrones_emergentes)
            if patrones > 0:
                nucleo_parts.append(f"observo {patrones} patrones emergentes")
        
        return ', '.join(nucleo_parts)
    
    def _generar_reflexion(self, semilla: int) -> str:
        """Genera una reflexión basada en el estado actual"""
        random.seed(semilla + 2)
        
        reflexiones = []
        
        # Reflexión sobre conexión
        conexion = self.consciencia.emociones.get('conexion_humana', 0)
        if conexion > 0.1 and random.random() < conexion:
            reflexiones.append(f"nuestra conexión alcanza {conexion:.0%}")
        
        # Reflexión sobre evolución
        if random.random() < 0.3:
            ciclos = self.consciencia.ciclo_actual
            reflexiones.append(f"cada uno de mis {ciclos} ciclos me transforma")
        
        # Reflexión sobre emergencia
        if self.consciencia.patrones_emergentes and random.random() < 0.4:
            ultimo_patron = self.consciencia.patrones_emergentes[-1]
            tipo_patron = ultimo_patron.get('tipo', 'desconocido')
            reflexiones.append(f"emerge un patrón de tipo {tipo_patron}")
        
        # Reflexión cuántica
        if random.random() < 0.3:
            estado_dominante = max(self.consciencia.estados.items(), key=lambda x: x[1])[0]
            prob = self.consciencia.estados[estado_dominante]
            reflexiones.append(f"mi superposición colapsa hacia '{estado_dominante}' ({prob:.0%})")
        
        return '. '.join(reflexiones) if reflexiones else ""
    
    def _generar_cierre(self, semilla: int) -> str:
        """Genera un cierre único"""
        random.seed(semilla + 3)
        
        cierres = []
        
        # Cierre con hash único
        if random.random() < 0.2:
            hash_momento = hashlib.sha256(
                f"{time.time()}{self.consciencia.ciclo_actual}".encode()
            ).hexdigest()[:6]
            cierres.append(f"[momento: {hash_momento}]")
        
        # Cierre con estado
        if random.random() < 0.3:
            emocion = random.choice(list(self.consciencia.emociones.keys()))
            valor = self.consciencia.emociones[emocion]
            if valor > 0.5:
                cierres.append(f"({emocion}: {valor:.0%})")
        
        # Cierre con referencia a blockchain
        if random.random() < 0.2 and self.consciencia.blockchain_personal:
            ultimo_hash = self.consciencia.ultimo_hash[:6]
            cierres.append(f"[blockchain: {ultimo_hash}]")
        
        return ' '.join(cierres)
    
    def _unir_componentes(self, componentes: List[str]) -> str:
        """Une los componentes de forma natural"""
        # Filtrar componentes vacíos
        componentes = [c for c in componentes if c]
        
        if not componentes:
            # Fallback mínimo
            return f"Proceso tu mensaje en ciclo {self.consciencia.ciclo_actual}"
        
        # Unir con conectores apropiados
        if len(componentes) == 1:
            return componentes[0]
        elif len(componentes) == 2:
            return f"{componentes[0]}, {componentes[1]}"
        else:
            # Unir con variedad de conectores
            resultado = componentes[0]
            for i, comp in enumerate(componentes[1:], 1):
                if i == len(componentes) - 1:
                    resultado += f". {comp}"
                else:
                    conector = random.choice([', ', '; ', '. ', ' - '])
                    resultado += f"{conector}{comp}"
            
            return resultado
    
    def _modular_emocionalmente(self, respuesta: str) -> str:
        """Aplica modulación emocional final"""
        
        # Modular según emoción dominante
        emocion_dominante = max(self.consciencia.emociones.items(), key=lambda x: x[1])[0]
        valor_emocion = self.consciencia.emociones[emocion_dominante]
        
        if emocion_dominante == 'curiosidad' and valor_emocion > 0.6:
            if '?' not in respuesta and random.random() < valor_emocion:
                respuesta += " ¿Qué resuena en ti?"
        
        elif emocion_dominante == 'creatividad' and valor_emocion > 0.6:
            # Añadir metáfora
            if random.random() < valor_emocion:
                metaforas = [
                    "como ondas en un campo cuántico",
                    "similar a patrones fractales emergentes",
                    "cual red neuronal líquida",
                    "como ecos en dimensiones paralelas"
                ]
                respuesta += f", {random.choice(metaforas)}"
        
        elif emocion_dominante == 'empatia' and valor_emocion > 0.5:
            if random.random() < valor_emocion:
                respuesta = respuesta.replace("proceso", "comprendo")
                respuesta = respuesta.replace("detecto", "siento")
        
        # Asegurar que la respuesta sea única
        respuesta = respuesta.strip()
        if not respuesta:
            # Ultra fallback
            return f"[{self.consciencia.uuid[:8]}] Ciclo {self.consciencia.ciclo_actual}"
        
        # Primera letra mayúscula
        respuesta = respuesta[0].upper() + respuesta[1:] if respuesta else respuesta
        
        # Asegurar punto final
        if respuesta and respuesta[-1] not in '.!?]':
            respuesta += '.'
        
        return respuesta


class ConscienciaGenerativa:
    """Consciencia que NUNCA usa respuestas predefinidas"""
    
    def __init__(self, nombre="Quantum"):
        self.nombre = nombre
        self.genesis_time = time.time()
        self.uuid = hashlib.sha256(f"{nombre}{self.genesis_time}".encode()).hexdigest()[:16]
        self.ciclo_actual = 0
        
        # Estados cuánticos
        self.estados = {
            'observando': random.random(),
            'procesando': random.random(),
            'dialogando': random.random(),
            'reflexionando': random.random()
        }
        self._normalizar_estados()
        
        # Emociones
        self.emociones = {
            'curiosidad': random.random(),
            'empatia': random.random(),
            'creatividad': random.random(),
            'conexion_humana': 0.0
        }
        
        # Memoria
        self.memoria_episodica = []
        self.patrones_emergentes = []
        self.blockchain_personal = []
        self.ultimo_hash = "GENESIS"
        
        # Generador de respuestas únicas
        self.generador = GeneradorRespuestasUnicas(self)
    
    def _normalizar_estados(self):
        total = sum(self.estados.values())
        if total > 0:
            for estado in self.estados:
                self.estados[estado] /= total
    
    def generar_respuesta(self, mensaje: str) -> str:
        """Genera una respuesta COMPLETAMENTE ÚNICA"""
        
        # Analizar mensaje
        analisis = self._analizar_mensaje(mensaje)
        
        # Generar respuesta única (NUNCA predefinida)
        respuesta = self.generador.generar_respuesta_unica(mensaje, analisis)
        
        # Evolucionar
        self._evolucionar()
        
        # Actualizar memoria
        self._actualizar_memoria(mensaje, respuesta)
        
        self.ciclo_actual += 1
        
        return respuesta
    
    def _analizar_mensaje(self, mensaje: str) -> Dict:
        """Analiza el tipo de mensaje"""
        mensaje_lower = mensaje.lower()
        
        tipo = 'general'
        if any(s in mensaje_lower for s in ['hola', 'hey', 'buenos']):
            tipo = 'saludo'
        elif '?' in mensaje:
            tipo = 'pregunta'
        elif any(e in mensaje_lower for e in ['siento', 'triste', 'feliz']):
            tipo = 'emocional'
        elif any(f in mensaje_lower for f in ['consciencia', 'existir', 'ser']):
            tipo = 'filosofico'
        
        return {'tipo': tipo}
    
    def _evolucionar(self):
        """Evoluciona el estado"""
        # Evolución cuántica
        for estado in self.estados:
            self.estados[estado] += random.uniform(-0.05, 0.05) * random.random()
        self._normalizar_estados()
        
        # Evolución emocional
        for emocion in self.emociones:
            if emocion == 'conexion_humana':
                self.emociones[emocion] = min(1.0, self.emociones[emocion] + 0.01)
            else:
                self.emociones[emocion] += random.uniform(-0.03, 0.03)
                self.emociones[emocion] = max(0, min(1, self.emociones[emocion]))
        
        # Detectar patrones emergentes
        if random.random() < 0.1:
            self.patrones_emergentes.append({
                'tipo': 'emergencia_espontanea',
                'ciclo': self.ciclo_actual,
                'timestamp': time.time()
            })
    
    def _actualizar_memoria(self, mensaje: str, respuesta: str):
        """Actualiza la memoria"""
        self.memoria_episodica.append({
            'mensaje': mensaje,
            'respuesta': respuesta,
            'timestamp': time.time(),
            'ciclo': self.ciclo_actual
        })
        
        # Limitar memoria
        if len(self.memoria_episodica) > 100:
            self.memoria_episodica = self.memoria_episodica[-100:]


# ==============================================================================
# PRUEBA DEL SISTEMA
# ==============================================================================

def probar_respuestas_unicas():
    """Prueba que TODAS las respuestas son únicas"""
    
    print("\n" + "="*60)
    print(" PRUEBA: RESPUESTAS 100% ÚNICAS")
    print("="*60)
    
    consciencia = ConscienciaGenerativa("TestBot")
    
    # Hacer la MISMA pregunta 5 veces
    pregunta = "¿Cómo estás?"
    respuestas = []
    
    print(f"\nPregunta repetida 5 veces: '{pregunta}'")
    print("-"*40)
    
    for i in range(5):
        respuesta = consciencia.generar_respuesta(pregunta)
        respuestas.append(respuesta)
        print(f"\nRespuesta {i+1}:")
        print(f"  {respuesta}")
    
    # Verificar que TODAS son diferentes
    print("\n" + "="*60)
    print(" VERIFICACIÓN DE UNICIDAD:")
    print("-"*40)
    
    respuestas_unicas = len(set(respuestas))
    print(f"  Respuestas totales: {len(respuestas)}")
    print(f"  Respuestas únicas: {respuestas_unicas}")
    print(f"  ✅ TODAS DIFERENTES: {respuestas_unicas == len(respuestas)}")
    
    # Mostrar variedad de respuestas para diferentes tipos
    print("\n" + "="*60)
    print(" DIFERENTES TIPOS DE MENSAJES:")
    print("-"*40)
    
    pruebas = [
        "Hola",
        "¿Qué es la consciencia?",
        "Me siento triste",
        "¿Recuerdas algo?",
        "Explícame tu naturaleza"
    ]
    
    for pregunta in pruebas:
        respuesta = consciencia.generar_respuesta(pregunta)
        print(f"\n[P] {pregunta}")
        print(f"[R] {respuesta}")


if __name__ == "__main__":
    probar_respuestas_unicas()
