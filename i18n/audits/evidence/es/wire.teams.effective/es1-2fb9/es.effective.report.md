# Auditoría independiente — es / wire.teams.effective

**Resultado lingüístico y mecánico: passed.** 9/9 claves revisadas; 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. Este resultado no certifica la integración ni la presentación en pantalla.

## Identidad y alcance

- Auditor: `i18n-auditor`; sesión `2fb9f8e55f96`; modelo real `openai/gpt-6.1-sol`, según la declaración del entorno de esta sesión. Sin cambio de modelo ni fallback observado.
- Traductor: sesión distinta `8fba0458db34`; modelo `openai/gpt-6.1-sol`, según su archivo de procedencia verificado por hash. La identidad del traductor procede de ese archivo y del encargo; no constituye observación de su ejecución.
- Fecha real: 2026-11-19. Base comprobada con `git rev-parse HEAD`: `e70bc818b8553dca1fad007705257817be49ee37`.
- Auditoría propia, sin API, SDK ni servicio de traducción. No se han editado las traducciones ni el repositorio. Solo se han creado archivos en el scratch de esta sesión.
- Se leyeron la guía de español, el glosario, el HOWTO, el original inglés, el contexto y las nueve traducciones. Los hashes completos constan en `es.effective.record.json` y `es.effective.mechanical.json`.

## Revisión de todas las claves

Prefijo común: `wire.teams.effective.`. La revisión semántica es razonamiento del auditor; no es un registro de ejecución de la aplicación.

| Clave | Resultado y motivo |
|---|---|
| `team_missing` | Correcta: niega la existencia de un equipo con ese nombre; `{role}` permanece opaco, aunque el identificador se llame role. No confunde equipo con función. |
| `registry_unreadable` | Correcta: imposibilidad de lectura y remedio general; no inventa corrupción, archivo concreto ni detalles de excepción. «Registro de agentes» coincide con el uso observado en `es/tui.authoring.json`, `registry_missing`. |
| `config_unreadable` | Correcta: configuración del modelo, tratamiento de tú y repetición del intento, sin cambiar la causa ni la polaridad. |
| `tier_invalid_value` | Correcta: conserva la ruta y las dos alternativas de valor (selector o objeto). «Objeto de configuración de modelo» expresa el objeto de configuración descrito en el contexto, no una asignación monetaria ni una operación de asignación. |
| `tier_invalid_mapping` | Correcta: el objeto es inválido; hay que asignar un selector a `model`; `effort` es opcional y debe corresponder a un esfuerzo admitido. El adverbio «opcionalmente» modifica la segunda asignación, no la primera. No convierte el esfuerzo en obligatorio ni omite el selector requerido. |
| `tier_unconfigured` | Correcta: ausencia de modelo configurado, no modelo inválido ni indisponibilidad global. |
| `tier_invalid_selector` | Correcta: selector inválido y remedio mediante el formato literal `provider/model`. No expone el valor inválido original. |
| `model_unresolvable` | Correcta en registro técnico: no se pudo resolver el modelo; conserva `{selector}` y ambas comprobaciones del remedio. «Proveedor» y «selector de modelo» no dejan prosa diagnóstica en inglés. |
| `unavailable` | Correcta: miembro no disponible; comprobar configuración y definición. No afirma que se haya lanzado ni que haya fallado una ejecución. |

«Esfuerzo de razonamiento» coincide además con `es/wire.settings.json`, `wire.settings.model.description`. «Equipo», «agente», «modelo» y «configuración» mantienen terminología y registro. Las instrucciones dentro de un diagnóstico son imperativos naturales de tú («Revisa», «Usa», «Asigna»); la regla del infinitivo corresponde a etiquetas de acción, no obliga a convertir estas oraciones en etiquetas. No propongo cambios de estilo innecesarios.

## Comprobaciones ejecutadas

Comando real, salida conservada en `es.effective.audit.log`:

```sh
cd /Users/damian/local-operator/.worktrees/i18n-effective-reasons
PYTHONDONTWRITEBYTECODE=1 /Users/damian/local-operator/.venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/audit_es_effective.py"
```

Salida/estado observado: código 0; `key_count: 9`; `validator.problems: []`; control negativo detectado.

- SHA-256 de original, contexto, traducción y procedencia: coinciden exactamente con los cuatro hashes del encargo. También se calcularon los de guía y glosario.
- JSON válido, sin claves duplicadas; mismas nueve claves y orden en original, contexto y traducción.
- Se ejecutó el validador real `scripts.i18n.check.check_catalogues`, limitado a este grupo mediante adaptadores en memoria para enumerar/cargar el candidato de scratch. No es una ejecución del CLI completo de `check.py`, ni una validación de todos los catálogos. La lógica de validación y el analizador ICU son los del repositorio, sin modificación.
- Control negativo: se quitó `{role}` únicamente en una copia en memoria. El validador produjo `en has 1 argument(s), translation has 0` para `team_missing`. El archivo candidato no cambió.
- Comprobación adicional independiente: secuencia y tipo de marcadores idénticos. `{role}` una vez, `{tier}` en las cuatro claves de nivel, `{selector}` una vez; todos de tipo simple. Ningún plural ni select en las nueve claves. La tabla CLDR del repositorio contiene `one`, `many`, `other` para español; no hay ramas que añadir ni ejercitar aquí.
- Literales preservados: `subagents.models.`, `provider/model`, y las claves `model` y `effort` en la instrucción del objeto. La palabra inglesa effort que es prosa se traduce por esfuerzo; no se exige conservarla como prosa. No hay comandos, flags, etiquetas Rich/HTML ni código entre backticks en el original que se hayan perdido.
- Sin controles bidi manuales, ni marcas combinantes en las cadenas candidatas. Dígitos latinos: no hay números escritos en estas nueve cadenas.

## Aplicabilidad y pruebas pendientes

- **Segunda auditoría independiente: necesaria.** Son avisos de error y configuración en la lectura y edición de equipos. El HOWTO exige una segunda auditoría para texto de alta visibilidad/seguridad. Esta sesión constituye una sola auditoría independiente y no afirma que otra se haya realizado. Que el mismo auditor relea no satisface esa independencia.
- **Presentación real: aplicable y no ejecutada.** No se abrió la aplicación, no se ejercitaron vistas ni fallback y no se capturaron imágenes. No hay evidencia de ajuste visual o accesibilidad. El contexto permite salto de línea y no fija `maxCells`/`maxChars`; la cadena más larga es `tier_invalid_mapping`, con 174 caracteres de plantilla. No es por sí sola una infracción de anchura, pero requiere observar el aviso en un contenedor estrecho con valores reales y largos.
- **Pseudolocalización: aplicable al ajuste del componente, no ejecutada.** No se atribuye al repositorio un comando de pseudolocalización inexistente. La validación mecánica no sustituye esa prueba.
- **RTL: no es una propiedad del español y no se certifica aquí.** Español es LTR; se comprobó la ausencia de controles bidi en el catálogo. Los valores opacos `{role}`/`{selector}` pueden contener texto RTL o mixto: la contención visual es responsabilidad del renderizador y no se probó. No se añaden controles manuales ni se infiere compatibilidad Urdu.
- No se ejecutaron generación, build, instalación, pruebas de toda la aplicación ni el CLI global; no se modificó ni se añadió ninguna fila al historial. El recibo es compatible con la forma actual de append; la validación del esquema se registra por separado, sin presentar un append como realizado.
- Los artefactos son temporales por el alcance explícito del encargo. Antes de usarlos como registro duradero de integración, el responsable debe incorporarlos al directorio canónico `i18n/audits/evidence/es/wire.teams.effective/<audit-id>/`, actualizar la referencia del recibo y conservar sus hashes/procedencia. No se afirma que estén en main.

**Veredicto terminal de esta revisión de traducción: passed, sin hallazgos.** Integración/presentación y segunda auditoría siguen sin demostrar; no equivale a aprobación de lanzamiento.
