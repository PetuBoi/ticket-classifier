# Jak projekt funguje

Příkaz `train` načte CSV, ověří jeho strukturu i hodnoty a rozdělí požadavky podle
scénářů. Poté naučí pipeline, vyhodnotí ji a uloží model, metriky, matici záměn
a seznamy scénářů. Příkaz `predict` načte důvěryhodný model a z anglického textu
vrátí kategorii i odhadované pravděpodobnosti všech čtyř tříd. Argumenty zpracovává
`argparse`; chyby vstupů končí srozumitelnou zprávou a návratovým kódem 2.

## TF-IDF a logistická regrese

`TfidfVectorizer` převede text na číselný vektor. Používá jednotlivá slova a dvojice
sousedních slov. Častý výraz v konkrétním požadavku může získat vyšší váhu; výrazy
běžné napříč trénovacími dokumenty dostávají nižší váhu. Slovník a váhy IDF vznikají
pouze z trénovacích dat. Neznámé výrazy při predikci vektorizer ignoruje.

`LogisticRegression` se naučí váhy příznaků pro rozlišení kategorií
`account_access`, `hardware`, `network` a `software`. Výstupem jsou odhadované
pravděpodobnosti, jejichž součet je jedna. Nejvyšší určuje výslednou kategorii.
Nejde o jistotu ani o kalibrovanou spolehlivost. I nesouvisející text dostane jednu
ze čtyř kategorií. Pipeline spojuje převod textu a klasifikaci, takže při predikci
používá stejné zpracování jako při trénování. Kompletní pipeline ukládá `joblib`.
Načítat se smějí pouze modely z důvěryhodného zdroje: načtení může spustit kód.

## Rozdělení dat

Dataset obsahuje 192 syntetických požadavků: 48 na kategorii, vždy 16 scénářů
se třemi parafrázemi. `scenario_id` označuje stejný konkrétní problém. Celé scénáře
se rozdělí v poměru 75 : 25 se seedem 42 a se stratifikací podle kategorií.
Trénovací sada má 144 textů a testovací 48; obě obsahují všechny kategorie.
Scénáře se mezi sadami nepřekrývají. Tím omezujeme únik téměř totožných parafrází.
Přesné duplicity textů po ořezání okrajových mezer validace odmítá.

TF-IDF i klasifikátor se učí výhradně na trénovací sadě. Testovací sada slouží
k závěrečnému měření, nikoli k ladění parametrů. Uložené seznamy scénářů, seed
a `uv.lock` umožňují opakovat běh se stejnými daty a závislostmi.

## Metriky a omezení

Accuracy je podíl správně přiřazených požadavků. Precision říká, kolik predikcí
dané kategorie bylo správných; recall, kolik skutečných požadavků této kategorie
model našel. F1 je harmonický průměr precision a recall. Macro F1 průměruje F1
všech kategorií se stejnou vahou. Support je počet skutečných příkladů kategorie.
Matice záměn má skutečné třídy v řádcích a předpovězené ve sloupcích.

Baseline `DummyClassifier` vždy vybírá nejčastější trénovací kategorii. Poskytuje
jednoduché srovnání bez porozumění textu.

Výsledky na malém syntetickém datasetu nevyjadřují přesnost na firemních datech.
Kategorie se mohou překrývat a neznámá témata model nerozpoznává. Implementace,
dokumentace i syntetická data vznikly s pomocí AI (OpenAI Codex).
