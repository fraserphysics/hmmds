;;;
;;; hmmkeys.el - Convenient keybindings and settings for working with
;;;              the book's LaTeX sources.
;;;
;;; Copyright (C) 2007, 2025 by Andy Fraser <andy@fraserphysics.com>
;;;
;;; Time-stamp: "16-May-2007 10:16:25 karlheg"
;;;
;;; I (Andy) edited this in 2025 starting from Karl's work in 2007

;;; code:

(condition-case err
    (require 'reftex)
  (error err))


(defun observation-scalar (s time)
  "Fetch S and TIME from keyboard.  Put \ti{S}{T} in the buffer."
  (interactive "svariable:\nstime:")
  (insert  "\\ti{" s "}{" time "}"))

(define-key LaTeX-mode-map [f1] #'observation-scalar)



(defun observation-vector (s b e)
  "Fetch S, B, E from keyboard.  Put \ts{S}{B}{E} in the buffer."
  (interactive "svariable:\nsbegin:\nsend:")
  (insert  "\\ts{" s "}{" b "}{" e "}"))

(define-key LaTeX-mode-map [f2] #'observation-vector)



(defun typeP (sub arg)
  "Fetch SUB and ARG from keyboard.  Put P_{SUB}{ARG} in the buffer."
  (interactive "ssub:\nsarg:")
  (insert  "P_{" sub "} \\left(" arg " \\right)"))

(define-key LaTeX-mode-map [f3] 'typeP)

(add-hook 'LaTeX-mode-hook (lambda () (setq TeX-command-default "LaTeXmk")))

(setq-default TeX-output-dir "../../../build/TeX/book/")

(setq-default TeX-master "main")

(load-theme 'tango t)

(provide 'hmmkeys)
;;; hmmkeys.el ends here

;;;--------------------
;;; Local Variables:
;;; mode: emacs-lisp
;;; End:
