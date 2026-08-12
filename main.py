"""
Lead Optimization Service for NovoMCP
Molecule generation only — scaffold hopping & property-directed optimization.
ADMET/SA scoring is handled by addie-models and chem-props services via quanta-mcp.
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import uuid
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(
    format='[NovoMCP] %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('novomcp.lead-optimization')

app = FastAPI(
    title="NovoMCP Lead Optimization Service",
    description="Molecule generation via scaffold hopping and property optimization",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = int(os.getenv("PORT", "8023"))
API_KEY = os.getenv("LEAD_OPT_API_KEY", "")


async def validate_api_key(x_api_key: str = Header(None)):
    if not API_KEY:
        logger.warning("LEAD_OPT_API_KEY not configured — rejecting all requests")
        raise HTTPException(status_code=503, detail="Service not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class OptimizationRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of lead compound")
    target_properties: Dict[str, float] = Field(default_factory=dict)
    optimization_type: str = Field("scaffold_hop")
    max_variants: int = Field(10)
    n_candidates: Optional[int] = Field(None, description="Alias for max_variants (quanta-mcp compat)")

class ScaffoldHopRequest(BaseModel):
    smiles: str = Field(..., description="Input SMILES")
    max_hops: int = Field(10)

class OptimizeLeadsRequest(BaseModel):
    molecules: List[Dict[str, str]] = Field(..., description="List of molecules with id and smiles")
    optimization_type: str = Field("scaffold_hop")
    max_variants: int = Field(10)
    top_n: int = Field(5)


# ---------------------------------------------------------------------------
# Molecular Editor — scaffold hopping & property optimization
# ---------------------------------------------------------------------------

class MolecularEditor:
    # Ring replacement pairs for scaffold hopping — pre-parsed as RDKit Mol objects
    _RAW_REPLACEMENTS = [
        # --- Single 6-membered aromatic swaps ---
        ("c1ccccc1", "c1ccncc1", "benzene", "pyridine"),
        ("c1ccccc1", "c1ccncn1", "benzene", "pyrimidine"),
        ("c1ccccc1", "c1ncncn1", "benzene", "triazine"),
        # --- Single 5-membered aromatic swaps ---
        ("c1ccoc1", "c1ccnc1", "furan", "pyrrole"),
        ("c1ccoc1", "c1ccsc1", "furan", "thiophene"),
        ("c1ccsc1", "c1ccnc1", "thiophene", "pyrrole"),
        # --- 6→5 and 5→6 aromatic swaps (ring contraction/expansion) ---
        ("c1ccccc1", "c1ccoc1", "benzene", "furan"),
        ("c1ccccc1", "c1ccsc1", "benzene", "thiophene"),
        ("c1ccccc1", "c1cc[nH]c1", "benzene", "pyrrole"),
        # --- Single 6-membered saturated swaps ---
        ("C1CCCCC1", "C1CCNCC1", "cyclohexane", "piperidine"),
        ("C1CCCCC1", "C1CCOCC1", "cyclohexane", "tetrahydropyran"),
        ("C1CCCCC1", "C1CCNC1", "cyclohexane", "pyrrolidine"),
        ("C1CCCCC1", "C1CCSCC1", "cyclohexane", "thiane"),
        # --- 10π fused bicyclic aromatic swaps (naphthalene family) ---
        ("c1ccc2ccccc2c1", "c1ccc2ncccc2c1", "naphthalene", "quinoline"),
        ("c1ccc2ccccc2c1", "c1ccc2ncncc2c1", "naphthalene", "quinazoline"),
        ("c1ccc2ccccc2c1", "c1ccc2c(c1)ncnc2", "naphthalene", "quinazoline-isomer"),
        ("c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "naphthalene", "indole"),
        ("c1ccc2ccccc2c1", "c1ccc2occc2c1", "naphthalene", "benzofuran"),
        ("c1ccc2ccccc2c1", "c1ccc2sccc2c1", "naphthalene", "benzothiophene"),
        # --- Bioisosteric indole swaps ---
        ("c1ccc2[nH]ccc2c1", "c1ccc2occc2c1", "indole", "benzofuran"),
        ("c1ccc2[nH]ccc2c1", "c1ccc2sccc2c1", "indole", "benzothiophene"),
        ("c1ccc2[nH]ccc2c1", "c1ccc2ncnc2c1", "indole", "benzimidazole"),
        # --- Kinase scaffolds: quinazoline → bioisosteres (covers erlotinib,
        #     gefitinib, lapatinib, afatinib, vandetanib class) ---
        ("c1ccc2ncncc2c1", "c1ccc2ncccc2c1", "quinazoline", "quinoline"),
        ("c1ccc2ncncc2c1", "c1ccc2nccnc2c1", "quinazoline", "quinoxaline"),
        ("c1ccc2ncncc2c1", "c1ccc2ncnnc2c1", "quinazoline", "pyrido-triazine"),
        ("c1ccc2ncncc2c1", "c1ccc2nc[nH]c2c1", "quinazoline", "benzimidazole-2"),
        # --- Quinoline → bioisosteres (covers chloroquine, mefloquine,
        #     bedaquiline, camptothecin class) ---
        ("c1ccc2ncccc2c1", "c1ccc2nccnc2c1", "quinoline", "quinoxaline"),
        ("c1ccc2ncccc2c1", "c1ccc2[nH]ccc2c1", "quinoline", "indole"),
        ("c1ccc2ncccc2c1", "c1ccc2cnccc2c1", "quinoline", "isoquinoline"),
        # --- Pyrimidine → triazine / pyridazine (single-ring N-heterocycle
        #     swaps for kinase warheads) ---
        ("c1ccncn1", "c1ncncn1", "pyrimidine", "triazine"),
        ("c1ccncn1", "c1ccnnc1", "pyrimidine", "pyridazine"),
        ("c1ccncn1", "c1ccncc1", "pyrimidine", "pyridine"),
        # --- 14π tricyclic fused aromatic swaps (carbazole / acridine family) ---
        ("c1ccc2c(c1)[nH]c1ccccc12", "c1ccc2c(c1)oc1ccccc12", "carbazole", "dibenzofuran"),
        ("c1ccc2c(c1)[nH]c1ccccc12", "c1ccc2c(c1)sc1ccccc12", "carbazole", "dibenzothiophene"),
        ("c1ccc2c(c1)[nH]c1ccccc12", "c1ccc2nc3ccccc3cc2c1", "carbazole", "acridine"),
        ("c1ccc2nc3ccccc3cc2c1", "c1ccc2oc3ccccc3cc2c1", "acridine", "xanthene"),
        ("c1ccc2nc3ccccc3cc2c1", "c1ccc2[nH]c3ccccc3c2c1", "acridine", "phenoxazine-core"),
        # --- Saturated bicyclic swaps ---
        ("C1CCC2CCCCC2C1", "C1CCC2NCCCC2C1", "decalin", "aza-decalin"),
        ("C1CCC2CCCCC2C1", "C1CCC2OCCCC2C1", "decalin", "oxa-decalin"),
    ]

    REPLACEMENTS = []
    for _old_smi, _new_smi, _old_name, _new_name in _RAW_REPLACEMENTS:
        _old_mol = Chem.MolFromSmiles(_old_smi)
        _new_mol = Chem.MolFromSmiles(_new_smi)
        if _old_mol and _new_mol:
            REPLACEMENTS.append((_old_mol, _new_mol, _old_name, _new_name))

    def detect_scaffolds(self, smiles: str) -> List[str]:
        """Return list of known scaffold types detected in the molecule.
        Used for diagnostics when scaffold_hop finds no matches."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []
        detected = set()
        for old_mol, _new_mol, old_name, _new_name in self.REPLACEMENTS:
            if mol.HasSubstructMatch(old_mol):
                detected.add(old_name)
        return sorted(detected)

    @staticmethod
    def _skeleton_compatible(old_mol, new_mol) -> bool:
        """True when old/new share an identical heavy-atom skeleton (same atom
        count + same bond connectivity by index) and differ only in atom
        identities — i.e. the C<->N aromatic bioisostere pairs (benzene/pyridine,
        naphthalene/quinoline, quinazoline/quinoline/quinoxaline). For these the
        hop is a pure ring-atom element edit, so exocyclic substituents never
        have to move. Size-changing pairs (benzene->furan) return False and fall
        through to the existing replacement paths."""
        if old_mol.GetNumAtoms() != new_mol.GetNumAtoms():
            return False
        if old_mol.GetNumBonds() != new_mol.GetNumBonds():
            return False
        ob = set()
        for b in old_mol.GetBonds():
            ob.add((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            ob.add((b.GetEndAtomIdx(), b.GetBeginAtomIdx()))
        return all(
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) in ob
            for b in new_mol.GetBonds()
        )

    def _skeleton_swap(self, mol, old_mol, new_mol) -> List:
        """In-place ring-atom element swap for same-skeleton bioisosteres.

        Edits the matched ring atoms' atomic numbers (e.g. quinazoline N->C to
        make quinoline) on an RWMol copy, leaving every exocyclic bond intact —
        so substituted fused cores survive. This fixes erlotinib's
        4-anilinoquinazoline (anilino + two methoxyethoxy chains), which
        HasSubstructMatch hits but ReplaceSubstructs fragments into
        `C#Cc1cccc(N)c1.COCCO.COCCOc1ccc2ncccc2c1`. Products that can't
        sanitize/kekulize (chemically invalid swaps, e.g. an N forced onto a
        substituted ring position) are dropped."""
        products = []
        seen_local = set()
        for match in mol.GetSubstructMatches(old_mol, uniquify=True):
            rw = Chem.RWMol(mol)
            for old_idx, mol_idx in enumerate(match):
                target_num = new_mol.GetAtomWithIdx(old_idx).GetAtomicNum()
                atom = rw.GetAtomWithIdx(mol_idx)
                if atom.GetAtomicNum() != target_num:
                    atom.SetAtomicNum(target_num)
                    atom.SetNumExplicitHs(0)
                    atom.SetNoImplicit(False)
            try:
                product = rw.GetMol()
                Chem.SanitizeMol(product)
            except Exception:
                continue
            canon = Chem.MolToSmiles(product)
            if "." in canon or canon in seen_local:
                continue
            seen_local.add(canon)
            products.append(product)
        return products

    def scaffold_hop(self, smiles: str, max_hops: int = 10) -> List[Dict]:
        """Generate scaffold variations via RDKit substructure matching.
        Returns basic RDKit properties only —
        ADMET and SA scoring are handled downstream by addie-models and chem-props."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []

        variations = []
        seen = {Chem.MolToSmiles(mol)}

        # Pre-compute SMILES strings of the replacement Mols once, used by the
        # string-level substitution path below.
        replacement_smiles_cache = {}

        for old_mol, new_mol, old_name, new_name in self.REPLACEMENTS:
            if len(variations) >= max_hops:
                break
            if not mol.HasSubstructMatch(old_mol):
                continue

            try:
                # Primary path: string-level SMILES replacement.
                # Reliable for the common case (aspirin, benzene-containing
                # leads with simple substituents) because the input SMILES
                # often contains the literal ring substring (`c1ccccc1`).
                # Returns connected molecules without the fragmentation issue
                # that `AllChem.ReplaceSubstructs` produces on ring atoms with
                # multiple substituents (regressed in commit c46f48e when we
                # switched away from this approach).
                cache_key = id(old_mol)
                if cache_key not in replacement_smiles_cache:
                    replacement_smiles_cache[cache_key] = (
                        Chem.MolToSmiles(old_mol),
                        Chem.MolToSmiles(new_mol),
                    )
                old_smi, new_smi = replacement_smiles_cache[cache_key]

                candidate_products = []
                if old_smi and old_smi in smiles:
                    candidate_str = smiles.replace(old_smi, new_smi, 1)
                    candidate_mol = Chem.MolFromSmiles(candidate_str)
                    if candidate_mol is not None:
                        candidate_products.append(candidate_mol)

                # Skeleton-swap path: same-skeleton C<->N bioisosteres
                # (quinazoline->quinoline, benzene->pyridine, naphthalene->
                # quinoline, ...). Edits ring-atom elements in place so
                # substituents on substituted/fused cores stay attached — fixes
                # erlotinib and other fused leads the string path misses (their
                # quinazoline isn't a literal SMILES substring) and that
                # ReplaceSubstructs fragments.
                if not candidate_products and self._skeleton_compatible(old_mol, new_mol):
                    candidate_products.extend(self._skeleton_swap(mol, old_mol, new_mol))

                # Fallback path: AllChem.ReplaceSubstructs for cases where the
                # input's canonical SMILES doesn't contain the literal ring
                # substring (e.g., differently-canonicalized fused systems).
                # Filter `.`-fragmented outputs at the source.
                if not candidate_products:
                    replaced = AllChem.ReplaceSubstructs(mol, old_mol, new_mol)
                    for r in replaced:
                        try:
                            Chem.SanitizeMol(r)
                        except Exception:
                            continue
                        canon = Chem.MolToSmiles(r)
                        if "." in canon:
                            # Disconnected fragments — skip
                            continue
                        candidate_products.append(r)

                for product in candidate_products:
                    canonical = Chem.MolToSmiles(product)
                    if "." in canonical or canonical in seen:
                        continue
                    seen.add(canonical)

                    variations.append({
                        "smiles": canonical,
                        "modification": f"Replace {old_name} with {new_name}",
                        "mw": round(Descriptors.MolWt(product), 2),
                        "logp": round(Crippen.MolLogP(product), 2),
                        "hbd": Descriptors.NumHDonors(product),
                        "hba": Descriptors.NumHAcceptors(product),
                        "tpsa": round(Descriptors.TPSA(product), 2),
                        "rotatable_bonds": Descriptors.NumRotatableBonds(product),
                        "qed": round(QED.qed(product), 3),
                        "num_rings": Descriptors.RingCount(product),
                        "scaffold": Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(product)),
                    })

                    if len(variations) >= max_hops:
                        break
            except Exception as e:
                logger.debug(f"Scaffold hop failed for {old_name} → {new_name}: {e}")
                continue

        # Sort by QED (drug-likeness) descending
        variations.sort(key=lambda x: x["qed"], reverse=True)
        return variations

    def optimize_properties(self, smiles: str, target_properties: Dict) -> List[Dict]:
        """Property-directed optimization via scaffold hopping + target filtering."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return []

        candidates = self.scaffold_hop(smiles, max_hops=len(self.REPLACEMENTS))

        if not target_properties:
            return candidates

        for c in candidates:
            penalty = 0.0
            for prop, target in target_properties.items():
                actual = c.get(prop)
                if actual is not None:
                    penalty += abs(actual - target)
            c["target_penalty"] = round(penalty, 2)

        candidates.sort(key=lambda x: x.get("target_penalty", 999))
        return candidates


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------

mol_editor = MolecularEditor()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "lead-optimization", "version": "3.0.0", "port": PORT}

@app.get("/about")
async def about():
    return {
        "service": "Lead Optimization Service",
        "version": "3.0.0",
        "capabilities": [
            "Scaffold Hopping (ring replacements)",
            "Property-Directed Optimization",
            "Batch Lead Optimization"
        ],
        "note": "ADMET and SA scoring handled by addie-models and chem-props via quanta-mcp"
    }


@app.post("/scaffold-hop", dependencies=[Depends(validate_api_key)])
async def scaffold_hop(request: ScaffoldHopRequest):
    variations = mol_editor.scaffold_hop(request.smiles, request.max_hops)
    return {
        "input_smiles": request.smiles,
        "num_variations": len(variations),
        "variations": variations
    }


@app.post("/optimize", dependencies=[Depends(validate_api_key)])
async def optimize_molecule(request: OptimizationRequest):
    max_v = request.n_candidates or request.max_variants

    if request.optimization_type == "scaffold_hop":
        results = mol_editor.scaffold_hop(request.smiles, max_v)
    else:
        results = mol_editor.optimize_properties(request.smiles, request.target_properties)

    response = {
        "optimization_id": str(uuid.uuid4()),
        "input_smiles": request.smiles,
        "optimization_type": request.optimization_type,
        "num_variants": len(results),
        "variants": results
    }

    # Diagnostics when no variants generated — helps users understand why
    if not results and request.optimization_type == "scaffold_hop":
        detected = mol_editor.detect_scaffolds(request.smiles)
        if detected:
            response["diagnostic"] = (
                f"Detected scaffold(s): {', '.join(detected)}. "
                f"Substructure match succeeded but ReplaceSubstructs produced no "
                f"sanitizable products — likely due to fused ring constraints or "
                f"valence conflicts after substitution."
            )
        else:
            response["diagnostic"] = (
                "No editable scaffolds detected in input. Current generator supports: "
                "benzene, pyridine, pyrimidine, furan, thiophene, pyrrole, cyclohexane, "
                "piperidine, naphthalene, quinoline, indole, benzofuran, benzothiophene, "
                "carbazole, dibenzofuran, acridine, decalin. "
                "For unusual scaffolds, use MolMIM (optimize_molecule) for property-directed edits."
            )
    return response


@app.post("/optimize-leads", dependencies=[Depends(validate_api_key)])
async def optimize_leads(request: OptimizeLeadsRequest):
    job_id = str(uuid.uuid4())

    all_variants = []
    for mol in request.molecules:
        smiles = mol.get("smiles")
        mol_id = mol.get("id")

        if request.optimization_type == "scaffold_hop":
            variants = mol_editor.scaffold_hop(smiles, request.max_variants)
        else:
            variants = mol_editor.optimize_properties(smiles, {})

        for variant in variants:
            variant["parent_id"] = mol_id
            variant["parent_smiles"] = smiles
        all_variants.extend(variants)

    logger.info(f"Generated {len(all_variants)} variants for {len(request.molecules)} molecules")

    # Sort by QED and return top_n
    all_variants.sort(key=lambda x: x.get("qed", 0), reverse=True)
    top_candidates = all_variants[:request.top_n]

    return {
        "job_id": job_id,
        "status": "completed",
        "variants_generated": len(all_variants),
        "top_candidates": top_candidates,
        "summary": {
            "molecules_processed": len(request.molecules),
        }
    }


@app.get("/jobs/{job_id}/status", dependencies=[Depends(validate_api_key)])
async def get_job_status(job_id: str):
    return {
        "job_id": job_id,
        "status": "completed",
        "message": "Lead optimization runs synchronously in v3"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Lead Optimization Service v3.0.0 on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
