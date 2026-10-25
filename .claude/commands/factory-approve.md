---
description: Aprueba manualmente una feature marcada como needs-human y hace merge a main
argument-hint: <slug o rama de la feature>
---
Feature a aprobar: $ARGUMENTS

1. Localiza la rama (`feature/$ARGUMENTS` o la que coincida) y su `factory/decisions/<slug>.md`. Si no existe o no tiene `status: needs-human`, para y avísame.
2. Muéstrame el motivo de la escalada y `git diff main...<rama> --stat`.
3. En la rama, cambia la cabecera a `status: approved-by-human` y añade la fecha. Haz commit.
4. `git checkout main && git merge --no-ff <rama>` y ejecuta `scripts/check.sh`. Confirma que termina con `Todo correcto`.
5. Resume qué se ha integrado. No hagas `git push`.
