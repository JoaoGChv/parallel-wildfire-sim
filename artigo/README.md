# Artigo (SBC) — como usar no Overleaf

Esta pasta contém o artigo completo no padrão **SBC**, pronto para o Overleaf.

## Arquivos
- `artigo.tex` — o artigo (estrutura: Introdução, Formulação do Problema, Objetivos,
  Metodologia, Resultados, Conclusão), igual ao padrão do exemplo.
- `referencias.bib` — referências (BibTeX).
- `fig_camadas.png`, `fig_metrica.png`, `fig_online.png`, `fig_sensibilidade.png`,
  `fig_brasil.png` — figuras referenciadas no texto.

## Passo a passo (Overleaf)
1. Use o **template SBC** (o mesmo do artigo de exemplo já compilado): ele fornece
   `sbc-template.sty` e o estilo de bibliografia `sbc.bst`. A forma mais simples é
   **duplicar o projeto Overleaf do artigo anterior** (que já tem esses arquivos) e
   substituir o conteúdo por estes.
2. Faça upload de `artigo.tex`, `referencias.bib` e dos 5 `.png` desta pasta para a
   raiz do projeto (junto de `sbc-template.sty` e `sbc.bst`).
3. Defina `artigo.tex` como documento principal.
4. Compile com **pdfLaTeX** e a sequência: pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX
   (o Overleaf faz isso automaticamente ao recompilar).

## Observações
- O `.tex` usa `\usepackage{sbc-template}` e `\bibliographystyle{sbc}` — ambos vêm do
  template SBC; não é preciso editá-los.
- As figuras já têm os nomes exatos referenciados no `.tex` (`fig_*.png`).
- Os autores/afiliação estão preenchidos como no artigo de exemplo — ajuste se o
  grupo deste trabalho for diferente.
