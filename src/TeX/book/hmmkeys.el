;;;
;;; hmmkeys.el - Convenient keybindings and settings for working with
;;;              the book's LaTeX sources.
;;;
;;; Copyright (C) 2007 by Andy Fraser <andy@fraserphysics.com>
;;;
;;; Time-stamp: "16-May-2007 10:16:25 karlheg"
;;;

(require 'latex)
(condition-case err
    (require 'reftex)
  (error err))

(when (load "folding" 'nomessage 'noerror)
  (folding-mode-add-find-file-hook)
  (define-key global-map (kbd "<f4>")   #'folding-toggle-show-hide)
  (define-key global-map (kbd "S-<f4>") #'folding-open-buffer)
  (define-key global-map (kbd "C-<f4>") #'folding-whole-buffer)
  (define-key global-map (kbd "<f5>")   #'folding-toggle-enter-exit)
  (define-key global-map (kbd "S-<f5>") #'folding-show-all))

;;; For M-x copyright
;;;
(when (not (getenv "ORGANIZATION"))
  (setenv "ORGANIZATION" "Andrew Fraser <andy@fraserphysics.com>"))
;;;
(setq copyright-query 'function)
(add-hook 'write-file-hooks #'copyright-update)

(setenv "WFDB" (concat ". "
		       "./data/Apnea "
		       (getenv "HOME") "/wfdb-database "
		       "/usr/share/wfdb/database "
		       "http://www.physionet.org/physiobank/database"))

(setenv "WFDBCAL" "wfdbcal")

(setenv "TEXINPUTS_latex" ".:./SiamTeX//:./figs/:")
(setenv "BIBINPUTS" ".:")
(setenv "BSTINPUTS" ".:")

;;; AUCTeX doesn't seem to see those environment settings and so it is
;;; necessary to
;;;
;;; This is done with `add-to-list' so that it will be idempotent.  If
;;; the items are simply `cons' onto the front of the list, it will
;;; grow longer each time this file gets loaded.
;;;
(let ((dirs '("." "./SiamTeX/" "./figs/")))
  (setq TeX-check-path (remove-if #'(lambda (elt)
				      (string= "." elt))
				  TeX-check-path))
  (mapc #'(lambda (dir)
	    (add-to-list 'TeX-macro-private dir)
	    (add-to-list 'TeX-check-path dir))
	(reverse dirs)))

(add-to-list 'TeX-file-extensions "eps_t")

(add-to-list 'auto-mode-alist '("\\.eps_t\\'" . latex-mode))


;;; Get all of the files right away, ready to edit.
;;; `find-file-noselect' will not open a second copy of a file.  If
;;; it's already in a buffer, it only checks to make sure the file is
;;; not changed on disk.
;;;
(let ((enable-local-variables nil)
      (enable-local-eval nil))
  (mapc #'find-file-noselect
	`("main.tex"
	  "introduction.tex"
	  "algorithms.tex"
	  "variants.tex"
	  "continuous.tex"
	  "toys.tex"
	  "real.tex"
	  "appendix.tex"
	  "hmmds.bib"
	  "hmmdsbook.cls"
	  "Makefile"
	  "README"
	  "ToDo.txt"
	  )))



(setq enable-recursive-minibuffers t)


(defun observation-scalar (s time)
  ;; the function "interactive" fetches strings "variable" and "time"
  ;; from the keyboard
  (interactive "svariable:\nstime:")
  (insert  "\\ti{" s "}{" time "}"))

(define-key LaTeX-mode-map [f1] #'observation-scalar)




(defun observation-vector (s b e)
  (interactive "svariable:\nsbegin:\nsend:")
  (insert  "\\ts{" s "}{" b "}{" e "}"))

(define-key LaTeX-mode-map [f2] #'observation-vector)



(defun typeP (sub arg)
  (interactive "ssub:\nsarg:")
  (insert  "P_{" sub "} \\left(" arg " \\right)"))

(define-key LaTeX-mode-map [f3] 'typeP)
