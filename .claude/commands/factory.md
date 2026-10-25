---
description: Pipeline completo de la software factory
argument-hint: <descripción de la feature>
---
Feature solicitada: $ARGUMENTS

Antes de empezar, comprueba que el árbol de trabajo está limpio (`git status`) y que estás en `main`. Si no es así, para y avísame.

1. Usa el subagente spec-writer para escribir la spec en factory/specs/.
2. Crea una rama nueva `feature/<slug>` desde main, haz commit de la spec y usa el subagente builder para implementar la spec.
3. Lanza EN PARALELO (en un mismo mensaje) los subagentes security-review, ux-review,
   ui-design-review y code-review. Pásales la ruta de la spec y el nombre de la rama. Espera a que terminen los 4.
4. Si alguno devuelve `VERDICT: CHANGES`, pasa todo el feedback al builder y
   repite el paso 3. Máximo 2 rondas de cambios.
5. Si los 4 dan `VERDICT: PASS`, haz commit de los informes de factory/reviews/ y usa el subagente approver.
   - `DECISION: APPROVE` → `git checkout main && git merge --no-ff feature/<slug>` y comprueba que `scripts/check.sh` sigue terminando con `Todo correcto`.
   - `DECISION: ESCALATE`, o si se agotan las 2 rondas → guarda el motivo en
     `factory/decisions/<slug>.md` con la cabecera `status: needs-human`, haz commit en la rama y para.
     No hagas merge.

Nunca hagas `git push` ni envíos a Kaggle dentro del pipeline: eso lo decido yo después.

Al terminar, dame un resumen: la rama, el número de rondas, los veredictos de cada revisor y el resultado final.
