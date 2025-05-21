# Článek: *Highly accurate protein structure prediction with AlphaFold*  
**Článek a zdroj**: *Jumper et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature.*  [Odkaz](https://www.nature.com/articles/s41586-021-03819-2)

---------------------------------------------------------------------------------------------------------------------------------------------

## Úvod:

Predikce 3D struktury proteinů na základě jejich aminokyselinové sekvence je dlouhodobý výzkumný problém v biologii. V experimentech, jako je CASP (Critical Assessment of Structure Prediction), byly dosud používané metody často omezené v přesnosti. Cílem týmu DeepMind bylo vyvinout systém schopný předpovědět struktury s téměř experimentální přesností bez potřeby fyzikálních simulací.

---------------------------------------------------------------------------------------------------------------------------------------------

## Hlavní informace:

AlphaFold dosáhl průlomové přesnosti v soutěži **CASP14**, kde:
- dosáhl průměrného **GDT_TS skóre >90**, což znamená blízkost k experimentálně určeným strukturám.
- překonal všechny ostatní metody a v mnoha případech byl stejně přesný jako krystalografická data.

Systém je schopen modelovat i těžko dostupné proteiny bez existujících šablon.

---------------------------------------------------------------------------------------------------------------------------------------------

## Metodika a architektura AlphaFold:

AlphaFold využívá:

- **Evoformer**: modul, který integruje evoluční vztahy (MSA – multiple sequence alignment) a informace o párech reziduí.
- **Structure module**: výstupní síť, která předpovídá 3D pozice atomů a odhaduje přesnost modelu.
- **End-to-end trénink**: model je trénován přímo na predikci struktury, bez oddělené fáze shlukování nebo optimalizace.
- **Predikční skóre (pLDDT)**: odhad lokální přesnosti predikce (0–100), které se ukazuje jako spolehlivé.

Tréninkový dataset:
- Protein Data Bank (PDB), sekvenční databáze (Uniclust, BFD), použití evoluční informace.

---------------------------------------------------------------------------------------------------------------------------------------------

## Výsledky a benchmarky:

- **CASP14**: AlphaFold překonal ostatní týmy s obrovským rozdílem.
- V některých případech poskytl přesnější výsledky než existující experimentální metody.
- Predikce zahrnují i domény bez předchozí struktury a bez homologů.

---------------------------------------------------------------------------------------------------------------------------------------------

## Dílčí analýza:

- Modely s vysokým pLDDT skóre (>90) mají sub-atomickou přesnost.
- Protilátky, membránové proteiny a jiné obtížné cíle byly úspěšně predikovány.
- Na rozdíl od předchozích přístupů AlphaFold využívá vzdálenostní matice a kontaktové mapy zcela implicitně v rámci architektury.

---------------------------------------------------------------------------------------------------------------------------------------------

## Význam a dopad:

- Otevření AlphaFold Protein Structure Database (spolupráce s EMBL-EBI) → stovky tisíc volně dostupných struktur (včetně lidského proteomu).
- Usnadnění biologického výzkumu, lékařských aplikací, vývoje léčiv.
- Využití při strukturálním poznání COVID-19 proteinů.

---------------------------------------------------------------------------------------------------------------------------------------------

## Shrnutí a výhled:

AlphaFold představuje kvantový skok v bioinformatice a strukturální biologii. Představuje nový standard v predikci proteinových struktur. Přestože jde o statickou predikci, model lze kombinovat s dalšími nástroji (např. molekulární dynamikou) pro hlubší pochopení funkce proteinů.
