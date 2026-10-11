"""PDF-Export der aktuellen Auswertung (fpdf2, Kernschriften; Umlaute sind dort erlaubt, Gedankenstrich und Eurozeichen nicht)."""

from fpdf import FPDF

import bom_constants as C
import bom_evaluation as ev


def _num(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def build_pdf(a, choice):
    """Einseitiger Bericht: Einstellungen, Modellwahl, Regret von Zufall, Standard und Wahl. Gibt die Bytes der PDF-Datei zurück."""
    s = a.settings
    reg = ev.final_regret(a)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Das Ersatzmodell entscheidet", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sebastian Hanisch, Operations Research und Machine Learning", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    rows = [
        ("Landschaft", C.KIND_LABELS[a.kind]), ("Auswertungen", str(s.n)), ("Rauschen (Branin)", _num(s.noise, 0)), ("Zufalls-Seed", str(s.seed)),
        ("Kern", C.KERNEL_LABELS[choice.kernel]), ("Längenmaß", choice.lengthscale), ("Rauschannahme", choice.noise),
        ("Transformation", C.TRANSFORM_LABELS[choice.transform]), ("Akquisition", C.ACQ_LABELS[choice.acquisition] + (f", beta {_num(choice.beta)}" if choice.acquisition == "ucb" else "")),
    ]
    for k, v in rows:
        pdf.cell(60, 6, k)
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Regret der Empfehlung (wahrer Wert minus bester Wert der Landschaft)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for k, label in (("random", "Zufall"), ("default", "Standardmodell"), ("choice", "Gewähltes Modell")):
        pdf.cell(60, 6, label)
        pdf.cell(0, 6, _num(reg[k]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Ein einzelner Lauf zeigt nur, was bei diesem Seed geschah; die Studie der Demo mittelt über 20 feste Läufe.")
    return bytes(pdf.output())
