#!/usr/bin/env python3
"""Strict transitive axiom audits for concrete core proofs and full certification."""
import argparse,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from toolchain import lake
LAKE=lake()
THEOREMS=['modulus_prime','chunk_fits','signed_coefficients_fit',
'ParameterChecks.first_row','ParameterChecks.reciprocal_construction','ParameterChecks.red_count','ParameterChecks.green_count','ParameterChecks.blue_count','ParameterChecks.retained_count',
'Example.soundness','Example.completeness','Example.determinism','Example.missing_output_constraint_counterexample',
'Gadgets.boolean_soundness','Gadgets.boolean_completeness','Gadgets.linear_output',
'Gadgets.quintic_soundness','Gadgets.quintic_completeness','Gadgets.fused_horner','Gadgets.horner_append',
'Arithmetic.two_stages_eq_direct','Arithmetic.cast_firstStage','Arithmetic.cast_secondStage',
'Arithmetic.radix_packing_bound','Arithmetic.pixel_packing_bound','Arithmetic.chunk_packing_bound',
'Exported.row_count','Exported.challenge_alias','Exported.counter_row_present','Exported.counter_transition','Target.actual_challenge_preserved',
'Target.determinism_of_soundness']
# Core imports are explicit: the ordinary audit does not depend on the full-step
# acceptance module or silently omit modules missing from the foundation root.
CORE_MODULES = [
    'CircuitCorrectness.Seed',
    'CircuitCorrectness.ProgramCertificates.All',
    'CircuitCorrectness.DctProgramCertificates.All',
    'CircuitCorrectness.PackingCertificates',
    'CircuitCorrectness.HashTrace2.All',
    'CircuitCorrectness.HashTrace8.All',
    'CircuitCorrectness.HashNodeSound',
    'CircuitCorrectness.HashTree',
    'CircuitCorrectness.SatisfyingWitness',
    'CircuitCorrectness.Composition',
]
THEOREMS += [
    'Byte.soundness', 'Byte.complete_of_values',
    'ConcreteBytes.actual_prefix', 'ConcreteBytes.soundness',
    'ConcreteBytes.complete_of_values', 'ConcreteBytes.pixel_wire',
    'Seed.matches_iff', 'Seed.preserved',
    'ProgramCertificates.correct', 'ProgramCertificates.wellFormed',
    'ProgramCertificates.execution_satisfies', 'ProgramCertificates.execution_preserves',
    'DctProgramCertificates.firstProgram_satisfied',
    'DctProgramCertificates.hornerProgram_satisfied',
    'DctProgramCertificates.exported_dct_sound',
    'PackingCertificates.satisfied',
    'PoseidonProgram.hash_sound', 'HashTrace.checkpoints_sound',
    'HashTrace2.template_sound', 'HashTrace8.template_sound',
    'HashWiring.left_sat', 'HashWiring.right_sat', 'HashWiring.combine_sat',
    'HashWiring.sat48', 'HashWiring.sat49', 'HashWiring.sat50', 'HashWiring.sat51',
    'HashWiring.left', 'HashWiring.right', 'HashWiring.combine',
    'HashWiring.digest_left', 'HashWiring.digest_right',
    'HashWiring.digest_combine_value', 'HashWiring.chain_value',
    'HashTree.chain_sound', 'exists_satisfying_step', 'Target.connected_steps_of_soundness',
    # Integer interpretations are separate obligations, rather than premises of
    # the field-valued step theorem. Audit them explicitly as well.
    'DctSpec.coefficient_bounds', 'DctSpec.coefficient_global_bounds',
    'DctSpec.bounded_cast_injective', 'DctSpec.coefficient_cast_injective',
    'DctSpec.packedPixel_bound', 'DctSpec.packedChunk_bound',
    'DctSpec.packedChunk_field_bound', 'DctSpec.packedChunk_cast_value',
    'DctSpec.first_row_pruning', 'DctSpec.omitted_zero',
]
# These principal names and their exact propositions remain enforced by
# Certification.lean. Connected-step results are audited in addition to them.
FULL_THEOREMS = [
    'Target.step_soundness', 'Target.step_completeness', 'Target.step_determinism',
    'Target.connected_steps', 'Target.connected360',
]
ALLOWED = {'propext', 'Quot.sound', 'Classical.choice'}

def run(full=False):
    names = FULL_THEOREMS if full else THEOREMS
    imports = ['Certification'] if full else ['CircuitCorrectness', *CORE_MODULES]
    source = ''.join('import '+name+'\n' for name in imports)
    source += '\n'.join('#print axioms CircuitCorrectness.'+name for name in names)+'\n'
    path = ROOT / ('lean/AuditFull.lean' if full else 'lean/Audit.lean')
    path.write_text(source)
    (ROOT/'results').mkdir(exist_ok=True)
    output = ROOT / ('results/full_axiom_audit.json' if full else 'results/axiom_audit.json')
    result = {'status': 'running',
              'scope': 'full exported step and connected iterations' if full else 'concrete core proofs',
              'allowed_axioms': sorted(ALLOWED), 'theorems': []}
    def save():
        output.write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        p = subprocess.run([str(LAKE), 'env', 'lean', str(path)], cwd=ROOT/'lean',
                           capture_output=True, text=True)
        log = ROOT / ('results/full_axiom_dependencies.log' if full else 'results/axiom_dependencies.log')
        log.write_text(p.stdout+p.stderr)
        if p.returncode != 0:
            raise RuntimeError('Lean axiom audit failed; see '+str(log))
        entries = re.findall(r"'([^']+)' depends on axioms:\s*\[([^\]]*)\]", p.stdout, re.S)
        no_axioms = re.findall(r"'([^']+)' does not depend on any axioms", p.stdout)
        actual_names = [name for name, _ in entries]+no_axioms
        expected_names = {'CircuitCorrectness.'+name for name in names}
        if len(actual_names) != len(names) or set(actual_names) != expected_names:
            raise RuntimeError('Axiom audit did not report every expected theorem exactly once')
        for name, axioms in entries:
            used = {x.strip() for x in axioms.split(',') if x.strip()}
            if not used <= ALLOWED:
                raise RuntimeError(f'Unapproved axioms in {name}: {sorted(used-ALLOWED)}')
            result['theorems'].append({'theorem': name, 'axioms': sorted(used)})
        result['theorems'] += [{'theorem': name, 'axioms': []} for name in no_axioms]
        result['status'] = 'pass'
        save()
        return result
    except Exception as error:
        result['status'] = 'fail'
        result['reason'] = str(error)
        save()
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    result = run(args.full)
    print(json.dumps({'status': result['status'], 'theorems': len(result['theorems'])}))
