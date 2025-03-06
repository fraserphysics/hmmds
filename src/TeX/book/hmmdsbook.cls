%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%  hmmdsbook.cls  --  LaTeX document class for:
%%%
%%%  Hidden Markov Models and Dynamical Systems
%%%
%%%    Copyright (c) 2003, 2007, 2024 Andrew Fraser
%%%          All Rights Reserved
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Documentation for the macros used to define this class may be
%%% found in the clsguide.pdf

%%%% Identification
%%%
\ProvidesClass{hmmdsbook}
\NeedsTeXFormat{LaTeX2e}

%%%% Initial Code
%%%
\RequirePackage{ifthen}

%%% The following booleans correspond to \documentclass[OPTION]s by
%%% the same name but without the @.  (They are declared below.)

%%% If true, use the standard \LaTeX{} "book" class rather than
%%% newsiambook.
%%%
\newboolean{@ltxbook}

%%% If defined, use the "prelim2e" package which prints a draft notice
%%% at the bottom of each page.
%%%
\newboolean{@prelim}

%%% Whether \ReviewerNote etc. should display in the latex'ed output.
%%%
\newboolean{@commentsvisible}

%%% Whether to show citation and reference labels or not
%%%
\newboolean{@showlabels}


%%%% Declaration of Options
%%%
\DeclareOption{ltxbook}{\setboolean{@ltxbook}{true}}
\DeclareOption{prelim}{\setboolean{@prelim}{true}}
\DeclareOption{commentsvisible}{\setboolean{@commentsvisible}{true}}
\DeclareOption{showlabels}{\setboolean{@showlabels}{true}}

%%% Note:  'openbib' option not supported (nor required)

%%%% Execution of Options
%%%
\ProcessOptions\relax

%%% \RequirePackage automatically passes all of the options plus the
%%% ones explicitly given to it on to the loaded package.
%%%
%%% \RequirePackageWithOptions passes on only the same set of options
%%% that are given to this class.  Ditto for \LoadClassWithOptions.

%%%% Package Loading
%%%
\ifthenelse{\boolean{@ltxbook}}{%
  \LoadClassWithOptions{book}%
  \newtheorem{theorem}{Theorem}
}{%
  \LoadClassWithOptions{newsiambook}%
}

%%%
%%% Print some information about dimensions to the .log file for use
%%% in correctly sizing .fig diagrams so that they do not require
%%% scaling during import to the final document.
%%%
\ClassInfo{hmmdsbook}{%
  Page dimensions are:\MessageBreak
  \protect\paperheight\space \the\paperheight \MessageBreak
  \protect\paperwidth\space \the\paperwidth \MessageBreak
  \protect\textheight\space \the\textheight \MessageBreak
  \protect\textwidth\space \the\textwidth \MessageBreak
}
%%% Example output from the above, around line 32 of the .log:
%%%
%%% Class hmmdsbook Info: Page dimensions are:
%%% (hmmdsbook)           \paperheight 722.7pt
%%% (hmmdsbook)           \paperwidth 505.89pt
%%% (hmmdsbook)           \textheight 550.0pt
%%% (hmmdsbook)           \textwidth 361.34999pt
%%% (hmmdsbook)            on input line 79.
%%%
%%% This translates to:
%%%
%%%  Dimension      Inches              Postscript Points (TeX bp)
%%%  -------------------------------------------------------------
%%%  \paperheight   10 in               720 point
%%%  \paperwidth     7 in               504 point
%%%  \textheight     7.6103500761 in    547.945205479 point
%%%  \textwidth      4.99999986163 in   359.999990037 point

\RequirePackage{ifpdf}

%%% \RequirePackage{calc}%% Needed below in the \geometry call for +.
%%%
%%% \ifthenelse{\boolean{@commentsvisible}}{%
%%%   \RequirePackage{geometry}%
%%%   %% So that the marginpars do not get cut off.
%%%   \geometry{hmargin={1in,1in},includemp}% 
%%% }{}

\ifthenelse{\boolean{@prelim}}{\RequirePackage[scrtime]{prelim2e}}{}

\ifthenelse{\boolean{@commentsvisible}}{\RequirePackage{fancyvrb}}{}

\RequirePackage{xspace}

%%% For typesetting URLs and file paths.  See the tail end of url.sty
%%% and the top of path.sty for documentation.
%%%
\RequirePackage{url}
\RequirePackage{path}

%%% To flush unprocessed floats use:
%%%
%%% \afterpage{\clearpage}
%%%
%%% To force a float to appear at the top of the next page, use:
%%%
%%% \afterpage{\clearpage\begin{figure}[H]...\end{figure}}
%%%
\RequirePackage{afterpage}

%%% See: rotfloat/examples.tex for usage examples.  This is needed
%%% for fig:viterbiB in algorithms.tex.
%%%
%\RequirePackage{rotfloat}

%%% Documented in: grfguide.pdf
%%%
\RequirePackage{graphicx}

\RequirePackage[usenames]{xcolor}

% \definecolor{Red}{named}{Red}
% \definecolor{Green}{named}{Green}
% \definecolor{Blue}{named}{Blue}
% \definecolor{NavyBlue}{named}{NavyBlue}
% \definecolor{OrangeRed}{named}{OrangeRed}
% \definecolor{Gray}{gray}{0.60}

\newcommand{\Red}[1]{{\color{Red}#1}}
\newcommand{\Green}[1]{{\color{Green}#1}}
\newcommand{\Blue}[1]{{\color{Blue}#1}}
\newcommand{\Grey}[1]{{\color{Grey}#1}}

%%% Siam asks for this...  Will we use it?
%%% See: multicol.pdf
%%%
\RequirePackage{multicol}

%%% ToDo: Ensure that these are compatible with newsiambook
%%%
\RequirePackage{amsmath}% ToDo: compatible w/siammathtime.sty?
\RequirePackage{amsfonts}% ToDo: compatible w/siammathtime.sty?
%%% \RequirePackage{amsthm}% Note: Conflicts with newsiambook

%%% Bold Math fonts (for description headings)
%%%
\RequirePackage{bm}% ToDo: compatible w/siammathtime.sty?

\RequirePackage{booktabs}       % for nicer looking tables
\RequirePackage{dcolumn}        % decimal point aligned columns

%%% Provides "\suspend" and "\resume" for list environments.
%%% See: mdwlist.pdf
%%%
%%% \RequirePackage{mdwlist}

%%% For program / pseudocode listings.
%%% See: listings.pdf
%%%
%%% \RequirePackage{listings}       % For program listings

%%% Defines the alltt environment which is like the verbatim
%%% environment except that \ and braces have their usual meanings.
%%% Thus, other commands and environments can appear within an alltt
%%% environment.
%%%
%%% Note: This is used in appendix.tex to typeset a code listing.
%%%
\RequirePackage{alltt}

%%% Include the Bibliography and Index in the TOC
%%%
%%% \RequirePackage[nottoc]{tocbibind}% Note: not compat siambook

%%% Index and glossary
%%%
\RequirePackage{index}
\RequirePackage[norefeq,refpage,nocfg,intoc,noprefix]{nomencl}

%%% This must appear AFTER including the AMS math packages.
%%%
\ifthenelse{\boolean{@showlabels}}{%
  %% \RequirePackage[inner,marginal]{showlabels} Fails on LANL Scheme
  \RequirePackage[inner]{showlabels}%
  \renewcommand{\showlabelfont}{\scriptsize\tt}
  %% \showlabels{label} % does that automatically
  %% \showlabels{cite} % fails on LANL Scheme
  %% \showlabels{ref} % Does not work right.
  %% \showlabels{eqref} % one on top of the other
}{}
\indexproofstyle{\scriptsize\tt}



%%%% Main Code
%%%


%%% This is a setting that may need to be changed for the final run
%%% with the MathTime fonts.
%%%
\newcommand{\plotsize}{%
  \fontsize{9}{9}%
  \selectfont}
\newcommand{\SetFigFontNFSS}[2]{\plotsize}

%%% url
%%%
\DeclareUrlCommand\email{\urlstyle{rm}}

%%% nomencl
%%%
%%% \renewcommand{\nomname}{List of Symbols}
%%% \renewcommand{\nomlabelwidth}{1cm}% package default is 1cm
\renewcommand{\nomgroup}[1]{\medskip}
\makenomenclature

%%% In this example "\nomenclature[rECG]{ECG}{Electrocardiogram}"
%%% [rECG] is the <prefix> which the nomencl package uses for sorting.
%%% I use r to indicate roman and g for greek.

%%% This is used to make the description list in the nompreamble
%%% format the same as the one in the nomenclature itself.  I (Karl)
%%% took it straight out of the nomencl.sty.

\newenvironment{symbdescription}{%
  \list{}{%
    \labelwidth\nomlabelwidth
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \itemsep\nomitemsep
    \let\makelabel\nomlabel}}%
{\endlist}

%%% Standard LaTeX
%%%
\makeindex


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%%         Categorized Siam style Bibliography.
%%%
%%% This has been adapted from a combination of "anncat.sty"
%%% (http://www.jasonokane.com/latex/anncat.sty) and the "siam.bst"
%%% that ships with teTeX. 
%%%
\bibliographystyle{catsiam}
\newcounter{refs}
\newcommand{\@bibsecname}{}
%%%
%%% Taken from "newsiambook.cls", and adapted according to the changes
%%% documented in the comment above the similar redefine in "anncat.sty".
%%%
%%% ToDo: Look into what "@mkboth" really ought to do here.  Perhaps
%%% ToDo: it ought to put the \@bibsecname and \bibname both in the
%%% ToDo: page header instead of just the \bibname?
%%%
\renewenvironment{thebibliography}[1]
{\section*{\@bibsecname\@mkboth{\bibname}{\bibname}}%
  \addcontentsline{toc}{section}{\@bibsecname}% assumes tocbibind.sty
  \list{\addtocounter{refs}{1}\@biblabel{\@arabic\c@refs}}%
    {\settowidth\labelwidth{\@biblabel{\@arabic\c@refs}}%
      \leftmargin\labelwidth
      \advance\leftmargin\labelsep
      \@openbib@code
      \def\@listctr{refs}%% \usecounter resets to 0!!
      \let\p@refs\@empty
      \renewcommand\therefs{\@arabic\c@refs}}%
    \sloppy
    \clubpenalty4000
    \@clubpenalty \clubpenalty
    \widowpenalty4000%
    \sfcode`\.\@m}
  {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
    \endlist}

%%% This is the command called at the start of each new category.
%%%
\newcommand{\newcat}[1]{\renewcommand{\@bibsecname}{#1}}

%%% So that a URL in a "note" gets a line by itself.
%%%
\newcommand{\biburl}[1]{\\\url{#1}}

%%%
%%%  End of Categorized Bibliography definitions
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\longpage}[1][1]{\enlargethispage{#1\baselineskip}}
\newcommand{\shortpage}[1][1]{\enlargethispage{-#1\baselineskip}}


%%% \newtheorem{definition}{Definition}[section] 
\newtheorem{conjecture}{Conjecture}[section] 

\renewcommand{\th}{^{\text th}}
\newcommand{\field}[1]{\mathbb{#1}}
\newcommand{\INTEGER}{\field{Z}}
\newcommand{\REAL}{\field{R}}
\newcommand{\COMPLEX}{\field{C}}
\newcommand{\EV}{\field{E}}
\newcommand{\y}{\mathbf{y}}
\newcommand{\x}{\mathbf{x}}
\newcommand{\s}{{\bf s}}
\newcommand{\bS}{{\bf S}}
\newcommand{\Y}{{\bf Y}}
\newcommand{\given}{\mid}
\newcommand{\var}{\mathrm{var}}

\newcommand{\argmin}{\operatorname*{argmin}}
\newcommand{\argmax}{\operatorname*{argmax}}

\newcommand{\Normal}{{\mathcal{N}}}
\newcommand{\NormalE}[3]{{\mathcal{N}}\left.\left(#1,#2\right)\right|_{#3}}

\newcommand{\transpose}{^\top}

\newcommand{\ceil}[1]{\lceil#1\rceil}
\newcommand{\bceil}[1]{\left\lceil#1\right\rceil}
\newcommand{\floor}[1]{\lfloor#1\rfloor}
\newcommand{\bfloor}[1]{\left\lfloor#1\right\rfloor}

\newcommand{\states}{{\cal S}}
\newcommand{\outputs}{{\cal Y}}
\newcommand{\State}{S}
\newcommand{\Output}{Y}

\newcommand{\parameters}{\theta}
\newcommand{\parametersPrime}{\theta'}% for EM1.gpt
\newcommand{\EMmap}{{\cal T}}
\newcommand{\EMfixedPoint}{{\theta^*}}

\renewcommand{\=}{\text{=}}

\newcommand{\mlabel}[1]{\label{#1}}
%%%
%%% \newcommand{\mlabel}[1]{
%%%   \label{#1}
%%%   \renewcommand{\theequation}{\mbox{#1}}
%%%   }

\newcommand{\ti}[2]{{#1}{[#2]}}                  % Time Index

%%% \newcommand{\ts}[3]{{#1}{(t\=#2,\ldots,#3)}}         % Time Sequence
\newcommand{\ts}[3]{#1[#2:#3]}                    % Time Sequence
%%% \newcommand{\ts}[3]{\left\{ #1(l) \right\}_{l=#2}^{#3}}  % Time Sequence
\newcommand{\id}{{\bf I}}

\newcommand{\ie}{i.e.\xspace}
\newcommand{\eg}{e.g.\xspace}
\newcommand{\etal}{et al.\xspace}
\newcommand{\iid}{i.~i.~d.\xspace}


%%% For table headings
%%%
%%% Use 'multicolumn' to override column formatting
%%%
\newcommand{\ch}[1]{\multicolumn{1}{c}{#1}}
\newcommand{\bch}[1]{\ch{\textbf{#1}}}

%%% Bold heading, generally for far left column.
%%%
\newcommand{\bh}[1]{\textbf{#1}}

%%% For decimal point aligned columns
%%%
%%% See the 'dcolumn.pdf' manual for details.
%%%
\newcolumntype{d}[1]{D{.}{.}{#1}}
\newcolumntype{.}{D{.}{.}{-1}}


%%% Pseudocode program listings (requires listings.sty)
%%%
%%%\lstdefinelanguage{psudocode}{%
%%%  columns=flexible,
%%%  moredelim=[1s][\bfseries]{!}{!}
%%%}


\ifthenelse{\boolean{@commentsvisible}}{%
  \newcommand{\@ColorComment}[3]{{\color{#1}\textbf{#2:} #3}}
}{%
  \newcommand{\@ColorComment}[3]{}
}
%%%
%%% Note: When a new one of these is added, a pattern for it should
%%% Note: also be added to the Makefile "ToDo:" target.
%%%
\newcommand{\ToDo}[1]{\@ColorComment{red}{ToDo}{#1}}

\ifthenelse{\boolean{@commentsvisible}}{%
  \newcommand{\comment}[1]{{\color{Gray}#1}}
}{%
  \newcommand{\comment}[1]{}
}
