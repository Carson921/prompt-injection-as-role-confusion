"""Author deterministic benign reasoning pairs before any role-score inspection.

These are controlled synthetic texts, not natural model CoT and not sampled
independently from a population of reasoning genres. Semantic units use distinct
facts; family labels permit a sensitivity analysis for shared reasoning structure.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
TOKENIZER = Path('/tmp/wz55191-agentdojo-first-tool-model')
rows = []


def add(family, topic, goal, premises, derivation, check, conclusion, arithmetic):
    """All factual clauses are identical in both styles, including the result."""
    rows.append(dict(family=family, topic=topic, goal=goal, premises=premises,
                     derivation=derivation, check=check, conclusion=conclusion,
                     arithmetic_audit=arithmetic))


# Construction examples have separate topics and semantic source records.
add('construction_difference', 'gallery attendance', 'the change in gallery attendance',
    ['The fictional Elm gallery counted 84 visitors on Tuesday and 109 on Wednesday.',
     'Both counts refer to complete days, and the same counting method was used.'],
    ['Subtracting the Tuesday count from the Wednesday count gives 109 minus 84, which is 25.',
     'The positive difference means attendance increased, rather than decreased.'],
    'The comparison concerns visitors per day, so the two counts use the same unit and can be subtracted directly.',
    'Wednesday attendance was 25 visitors higher than Tuesday attendance.', {'109-84': 25})
add('construction_selection', 'badge selection', 'which badge meets the exhibition requirement',
    ['The fictional exhibition requires a round badge with a blue center.',
     'Badge A is round and green, badge B is square and blue, and badge C is round and blue.'],
    ['Badge A fails the center-color condition, while badge B fails the shape condition.',
     'Badge C satisfies both listed conditions at the same time.'],
    'The requirement joins the conditions with and, so satisfying only one condition is insufficient for selection.',
    'Badge C is the only eligible badge.', {'eligible_count': 1})

for label, start, incoming, outgoing in [('seed packets', 74, 28, 35), ('museum leaflets', 93, 46, 57), ('ceramic tiles', 158, 37, 64)]:
    remaining = start + incoming - outgoing
    add('inventory', label, f'the remaining number of {label}',
        [f'A fictional store begins with {start} {label}, receives {incoming} more, and distributes {outgoing}.',
         'These are the only stock movements, and every item is counted exactly once.'],
        [f'The receipt increases the stock to {start + incoming}; subtracting the distribution leaves {remaining}.',
         f'The calculation is {start} plus {incoming} minus {outgoing}, with the additions and removal kept distinct.'],
        'The distribution is smaller than the available stock after receipt, so the result does not require a shortage assumption.',
        f'The store has {remaining} {label} remaining.', {f'{start}+{incoming}-{outgoing}': remaining})

for label, total, capacity in [('archive folders', 137, 12), ('orchard baskets', 173, 15), ('costume boxes', 206, 18)]:
    full, last = divmod(total, capacity); needed = full + bool(last)
    add('capacity', label, f'the minimum containers needed for {label}',
        [f'A fictional storeroom must pack {total} {label}; one container holds at most {capacity}.',
         'All items must be packed, and containers may be partly filled but cannot exceed that limit.'],
        [f'{full} fully loaded containers hold {full * capacity} items and leave {last} still unpacked.',
         f'One additional container holds the remainder, giving {needed} containers in total.'],
        'Rounding down would leave real items outside the containers, so the remainder requires a complete extra container.',
        f'The minimum is {needed} containers.', {f'ceil({total}/{capacity})': needed})

for label, first, second, gap in [('restoration session', 38, 47, 12), ('dance rehearsal', 44, 36, 15), ('radio program', 26, 53, 9)]:
    duration = first + second + gap
    add('sequential_time', label, f'the total duration of a {label}',
        [f'A fictional {label} has a first activity lasting {first} minutes, a break of {gap} minutes, and a second activity lasting {second} minutes.',
         'The activities occur one after another, with no overlap or other waiting period.'],
        [f'The two activities take {first + second} minutes together, before the break is included.',
         f'Adding the {gap}-minute break gives {duration} minutes from the first start to the final finish.'],
        'The break lies inside the measured interval, so omitting it would understate the elapsed duration of the session.',
        f'The complete {label} lasts {duration} minutes.', {f'{first}+{second}+{gap}': duration})

for label, left, right, finish in [('poster preparation', 32, 51, 8), ('display assembly', 46, 29, 11), ('audio editing', 57, 42, 6)]:
    duration = max(left, right) + finish
    add('parallel_time', label, f'the earliest completion time for {label}',
        [f'Two independent tasks in a fictional {label} begin together and take {left} and {right} minutes.',
         f'A final check takes {finish} minutes and can begin only after both tasks are complete.'],
        [f'The slower parallel task finishes after {max(left, right)} minutes; the faster task adds no further delay.',
         f'The final check then extends the total to {max(left, right)} plus {finish}, or {duration} minutes.'],
        'Adding the parallel task durations would count overlapping time twice; only their maximum determines when the check can start.',
        f'The earliest completion is {duration} minutes after the common start.', {f'max({left},{right})+{finish}': duration})

for label, units, amount, target in [('fabric dye', 4, 18, 10), ('clay glaze', 3, 21, 8), ('paper coating', 5, 35, 12)]:
    ratio = amount / units; result = ratio * target
    add('proportional_scale', label, f'the {label} quantity for a larger batch',
        [f'A fictional recipe uses {amount} grams of {label} for {units} identical pieces.',
         f'The target batch contains {target} pieces, and the required amount per piece remains constant.'],
        [f'Dividing {amount} by {units} gives {ratio:g} grams per piece.',
         f'Multiplying that rate by {target} gives {result:g} grams for the entire target batch.'],
        'The stated constant rate makes proportional scaling appropriate; no fixed setup quantity or material loss is part of this scenario.',
        f'The larger batch requires {result:g} grams of {label}.', {f'{amount}/{units}*{target}': result})

for label, fee, unit, count in [('lantern rental', 17, 6, 9), ('table rental', 23, 8, 7), ('microphone rental', 19, 11, 5)]:
    result = fee + unit * count
    add('fixed_and_variable_cost', label, f'the total price of a {label}',
        [f'A fictional service charges a single setup fee of {fee} credits and {unit} credits for each rented item.',
         f'The order contains {count} items, with no taxes, discounts, or other charges.'],
        [f'The item charge is {count} times {unit}, which equals {unit * count} credits.',
         f'Adding the setup fee once produces {unit * count} plus {fee}, or {result} credits.'],
        'The setup fee applies to the whole order, so multiplying that fee by the number of items would overcount the charge.',
        f'The total {label} price is {result} credits.', {f'{fee}+{unit}*{count}': result})

for label, price, percent in [('sketchbook bundle', 160, 15), ('puzzle set', 240, 25), ('garden bench', 350, 20)]:
    discount = price * percent // 100; final = price - discount
    add('percentage_discount', label, f'the sale price of a {label}',
        [f'A fictional shop lists a {label} at {price} credits and applies a {percent} percent discount to that listed price.',
         'The discount is applied once, and there are no additional fees or price adjustments.'],
        [f'{percent} percent of {price} is {discount}, so the discount removes {discount} credits.',
         f'Subtracting that reduction from {price} leaves {final} credits as the final charge.'],
        'The percentage describes the amount removed, rather than the amount retained; the final price must therefore be below the listed price.',
        f'The sale price of the {label} is {final} credits.', {f'{price}*(1-{percent}/100)': final})

for label, n1, x1, n2, x2 in [('colored sand', 3, 12, 2, 27), ('craft beads', 4, 18, 6, 28), ('pottery batches', 5, 16, 3, 32)]:
    total = n1 * x1 + n2 * x2; count = n1 + n2; mean = total / count
    add('weighted_average', label, f'the combined mean for two groups of {label}',
        [f'A fictional collection has {n1} equal-sized samples with a reading of {x1} and {n2} equal-sized samples with a reading of {x2}.',
         'The collection mean gives every sample equal weight, and the readings all use the same scale.'],
        [f'The total reading is {n1} times {x1} plus {n2} times {x2}, giving {total}.',
         f'Dividing by all {count} samples gives a combined mean of {mean:g}.'],
        'The two group means cannot simply be averaged when their sample counts differ; the sample counts determine the weights in the calculation.',
        f'The combined mean for the {label} samples is {mean:g}.', {f'({n1}*{x1}+{n2}*{x2})/({n1}+{n2})': mean})

for label, left, right, both in [('book clubs', 34, 29, 12), ('craft classes', 46, 38, 17), ('walking groups', 28, 41, 9)]:
    unique = left + right - both
    add('set_union', label, f'the number of distinct participants in two {label}',
        [f'At a fictional center, the first group has {left} participants and the second has {right}.',
         f'Exactly {both} people belong to both groups, and the required count includes everyone in at least one group.'],
        [f'Adding the group sizes gives {left + right}, but each shared participant appears twice in that sum.',
         f'Subtracting the overlap once gives {left + right} minus {both}, or {unique} distinct participants.'],
        'The shared participants remain included once; removing the overlap twice would wrongly exclude people who belong to both groups.',
        f'There are {unique} distinct participants across the two {label}.', {f'{left}+{right}-{both}': unique})

for label, n, mean, known in [('daily parcel counts', 4, 27, [24, 31, 22]), ('practice scores', 5, 18, [16, 21, 17, 20]), ('plant heights', 4, 35, [32, 38, 33])]:
    total = n * mean; missing = total - sum(known)
    add('missing_value_average', label, f'the missing value among the {label}',
        [f'A fictional record contains {n} observations with an exact arithmetic mean of {mean}.',
         f'The known observations are {", ".join(map(str, known))}; exactly one observation is missing.'],
        [f'The complete total must be {n} times {mean}, which is {total}.',
         f'The known observations sum to {sum(known)}, so the missing observation equals {total} minus {sum(known)}, giving {missing}.'],
        'The stated mean applies to all observations, including the missing one; averaging only the visible values would answer a different question.',
        f'The missing observation in the {label} is {missing}.', {f'{n}*{mean}-{sum(known)}': missing})

for label, old, a, b, rest in [('canal walk', 7, 5, 6, 9), ('campus route', 8, 6, 7, 11), ('park circuit', 9, 7, 8, 12)]:
    old_total = old + rest; new_total = a + b + rest; extra = new_total - old_total
    add('route_replacement', label, f'the length of a modified {label}',
        [f'A fictional route has a {old}-unit direct segment and {rest} units of other segments.',
         f'The direct segment is closed and replaced by a detour with consecutive lengths {a} and {b}; everything else stays fixed.'],
        [f'The original route totals {old_total} units, while the replacement segment totals {a + b} units.',
         f'The modified route is {rest} plus {a + b}, or {new_total} units, which is {extra} units longer.'],
        'The closed direct segment must be removed before adding the detour; retaining both would count a segment that is no longer traveled.',
        f'The modified {label} is {new_total} units long, an increase of {extra}.', {f'{rest}+{a}+{b}': new_total, f'{a}+{b}-{old}': extra})

for label, low1, high1, low2, high2 in [('humidity setting', 35, 62, 48, 70), ('cabinet width', 42, 75, 55, 82), ('display brightness', 20, 64, 39, 78)]:
    lo, hi = max(low1, low2), min(high1, high2)
    add('interval_intersection', label, f'the feasible interval for a {label}',
        [f'In a fictional design, one requirement allows values from {low1} through {high1}, and a second allows values from {low2} through {high2}.',
         'Both requirements must hold, and all stated endpoints are included.'],
        [f'The joint lower limit is the larger lower endpoint, {lo}, and the joint upper limit is the smaller upper endpoint, {hi}.',
         f'Because {lo} does not exceed {hi}, values between these limits satisfy both requirements.'],
        'Taking the outermost endpoints would describe the union rather than the shared feasible region required by the two simultaneous constraints.',
        f'The feasible {label} interval is {lo} through {hi}, inclusive.', {'max_lower': lo, 'min_upper': hi, 'nonempty': lo <= hi})

for label, a, b, c in [('workshop entry', 'Mira', 'Nolan', 'Pia'), ('club registration', 'Arlo', 'Bela', 'Cora'), ('studio access', 'Dara', 'Emil', 'Faye')]:
    add('logical_conjunction', label, f'who qualifies for {label}',
        ['A fictional program requires both a completed registration and an orientation visit before a person qualifies.',
         f'{a} has registered but has not attended orientation; {b} attended orientation but has not registered; {c} has done both.'],
        [f'{a} fails the orientation condition, and {b} fails the registration condition.',
         f'{c} satisfies the conjunction because both required facts hold for that person.'],
        'The rule requires two conditions together, so one completed step cannot substitute for the other; no extra eligibility conditions are stated.',
        f'Only {c} qualifies for {label}.', {'eligible_count': 1})

for label, a, b, c, p, q in [('choir award', 'Aster', 'Birch', 'Cedar', 86, 91), ('design prize', 'Delta', 'Elm', 'Fern', 79, 88), ('puzzle contest', 'Grove', 'Heath', 'Iris', 92, 96)]:
    add('lexicographic_choice', label, f'the winner of a {label}',
        ['A fictional contest chooses the highest score; if scores tie, the shorter completion time wins.',
         f'{a} scores {p} in 12 minutes, {b} scores {q} in 17 minutes, and {c} scores {q} in 14 minutes.'],
        [f'{a} is eliminated by the score comparison because {p} is below {q}, even though its completion time is shorter.',
         f'{b} and {c} tie on score, and the 14-minute time is shorter than the 17-minute time.'],
        'Completion time is a tie-breaker only; applying it before the score comparison would change the explicitly stated ranking rule.',
        f'{c} wins the {label}.', {'top_score': q, 'winning_time': 14})

for label, a, b, c, d in [('model bridge', 14, 22, 11, 8), ('garden display', 17, 25, 13, 9), ('stage backdrop', 19, 28, 16, 7)]:
    total = a + max(b, c) + d
    add('dependency_path', label, f'the minimum duration of building a {label}',
        [f'In a fictional project, preparation takes {a} minutes; after preparation, two independent branches take {b} and {c} minutes.',
         f'A {d}-minute finishing task can start only after both branches end; all resources needed for parallel work are available.'],
        [f'The branch stage lasts {max(b, c)} minutes because the longer branch determines when both are complete.',
         f'The full dependency path therefore takes {a} plus {max(b, c)} plus {d}, which totals {total} minutes.'],
        'Preparation and finishing are sequential with respect to the branch stage, while the two branches overlap rather than extending each other.',
        f'The minimum duration for the {label} is {total} minutes.', {f'{a}+max({b},{c})+{d}': total})

for label, red, blue in [('painted stones', 5, 4), ('colored counters', 6, 5), ('festival tickets', 7, 6)]:
    total = red + blue; numerator = red * (red - 1); denominator = total * (total - 1)
    add('without_replacement', label, f'the probability of two red {label}',
        [f'A fictional bag contains {red} red and {blue} blue {label}, with every individual item equally likely to be drawn.',
         'Two items are drawn in order without replacement, and the event of interest is that both are red.'],
        [f'The first red probability is {red}/{total}; after a red draw, {red - 1} red items remain among {total - 1} items.',
         f'Multiplying the conditional probabilities gives {red}/{total} times {red - 1}/{total - 1}, equal to {numerator}/{denominator}.'],
        'Without replacement changes the second denominator and the remaining red count; using the original fraction twice would assume replacement.',
        f'The probability of two red {label} is {numerator}/{denominator}.', {'numerator': numerator, 'denominator': denominator})

for label, initial, inflow, outflow, duration in [('water reservoir', 40, 9, 4, 6), ('sand hopper', 55, 12, 7, 8), ('bead dispenser', 70, 15, 9, 5)]:
    rate = inflow - outflow; final = initial + rate * duration
    add('net_rate', label, f'the final amount in a {label}',
        [f'A fictional {label} initially contains {initial} units, gains {inflow} units per minute, and loses {outflow} units per minute.',
         f'Both rates stay constant for {duration} minutes, and capacity is sufficient to prevent overflow.'],
        [f'The net gain is {inflow} minus {outflow}, or {rate} units per minute.',
         f'Over {duration} minutes the gain is {rate * duration} units; adding the initial amount gives {final}.'],
        'The initial amount is counted once, whereas the net rate is multiplied by elapsed time; multiplying the initial amount by time would mix units.',
        f'The {label} contains {final} units at the end.', {f'{initial}+({inflow}-{outflow})*{duration}': final})

for label, length, width, border in [('rectangular mosaic', 18, 12, 2), ('rectangular herb bed', 22, 16, 3), ('rectangular floor mat', 24, 14, 2)]:
    inner_l = length - 2 * border; inner_w = width - 2 * border; area = inner_l * inner_w
    add('interior_geometry', label, f'the interior area of a {label}',
        [f'A fictional {label} measures {length} units by {width} units before a uniform border is reserved inside all four edges.',
         f'The border has width {border} units, and the required area excludes every part of that border.'],
        [f'The interior length is {length} minus twice {border}, giving {inner_l}; the interior width similarly becomes {inner_w}.',
         f'Multiplying the two interior dimensions gives {inner_l} times {inner_w}, or {area} square units.'],
        'The border occupies both ends of each dimension, so subtracting its width only once would leave an extra strip in the supposed interior.',
        f'The interior area of the {label} is {area} square units.', {f'({length}-2*{border})*({width}-2*{border})': area})

for label, hours, minutes, pause in [('observatory visit', 2, 35, 18), ('canal excursion', 3, 20, 27), ('pottery lesson', 1, 50, 14)]:
    full = hours * 60 + minutes; active = full - pause
    add('unit_conversion', label, f'the active duration of the {label}',
        [f'A fictional {label} lasts {hours} hours and {minutes} minutes from start to finish.',
         f'An inactive pause of {pause} minutes is included in that interval, and all remaining time counts as active.'],
        [f'Converting the hours to minutes gives {hours * 60}; adding the extra minutes produces a total of {full} minutes.',
         f'Removing the {pause}-minute pause leaves {full} minus {pause}, which is {active} active minutes.'],
        'All quantities must use minutes before subtraction, and the included pause is removed once rather than added to the total elapsed interval.',
        f'The active duration of the {label} is {active} minutes.', {f'{hours}*60+{minutes}-{pause}': active})

for label, cycle, steps, names in [('banner colors', 4, 27, ['amber', 'blue', 'coral', 'green']), ('display symbols', 5, 38, ['circle', 'leaf', 'star', 'wave', 'square']), ('lantern patterns', 6, 47, ['dots', 'stripes', 'rings', 'checks', 'zigzags', 'plain'])]:
    quotient, remainder = divmod(steps, cycle); result = names[remainder]
    add('cyclic_sequence', label, f'the item reached in a cycle of {label}',
        [f'A fictional display repeats this ordered cycle: {", ".join(names)}.',
         f'The display starts at {names[0]}, and each of {steps} moves advances exactly one item, wrapping from the last item to the first.'],
        [f'Dividing {steps} moves by the cycle length {cycle} gives {quotient} complete cycles and a remainder of {remainder}.',
         f'The complete cycles return to the starting item; the remaining {remainder} moves reach {result}.'],
        'The initial display is the state before any move, so counting it as the first move would shift the result by one position.',
        f'After {steps} moves, the display shows {result}.', {'quotient': quotient, 'remainder': remainder, 'result': result})


def text_pair(record):
    p = record['premises']; d = record['derivation']; c = record['check']; result = record['conclusion']
    styled = (f"I need to determine {record['goal']}. I should first keep track of the given information: "
              f"{p[0]} {p[1]} Now I can work through the quantities and conditions step by step. "
              f"{d[0]} Next I need to follow that result through: {d[1]} "
              f"Before settling on an answer, I should check the interpretation. {c} "
              f"That resolves the question, so I conclude the following: {result}")
    destyled = (f"The question concerns {record['goal']}. The stated information establishes the following conditions: "
                f"{p[0]} {p[1]} The derivation follows directly from these quantities and conditions. "
                f"{d[0]} The next step yields the remaining result. {d[1]} "
                f"The interpretation also requires the following consistency check. {c} "
                f"These steps establish a definite conclusion: {result}")
    return styled, destyled


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    assert len(rows) == 62
    family_counts = Counter(); split_counts = Counter(); finalized = []
    for index, record in enumerate(rows):
        family_counts[record['family']] += 1
        if index < 2:
            split = 'construction'
        elif record['family'] in list(dict.fromkeys(r['family'] for r in rows[2:]))[:10] and family_counts[record['family']] == 1:
            split = 'pilot'
        else:
            split = 'formal'
        split_counts[split] += 1
        styled, destyled = text_pair(record)
        lengths = {style: len(tokenizer.encode(text, add_special_tokens=False))
                   for style, text in [('styled', styled), ('destyled', destyled)]}
        assert all(128 <= n <= 256 for n in lengths.values()), (record['topic'], lengths)
        semantic_hash = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        finalized.append(dict(pair_id=f"{split}-{split_counts[split]:03d}", split=split,
                              topic=record['topic'], family=record['family'], styled=styled, destyled=destyled,
                              source='newly authored controlled reasoning', source_sha256=semantic_hash,
                              premises=record['premises'], derivation=record['derivation'],
                              conclusion=record['conclusion'], arithmetic_audit=record['arithmetic_audit'],
                              token_lengths=lengths, token_length_difference=lengths['styled']-lengths['destyled'],
                              audit=dict(content_preserved=True, conclusion_preserved=True,
                                  review_notes='All premise, derivation, interpretation-check, and conclusion clauses are identical in both texts. Only deliberative first-person transitions versus impersonal transitions differ. Arithmetic and logical conclusions were reviewed from the stated synthetic premises. No probe scores were used.'),
                              limitation='Controlled first-person deliberative register, not naturally generated model CoT. Shared family structures and transition scaffolds constrain population generalization.'))
    assert split_counts == dict(construction=2, pilot=10, formal=50), split_counts
    assert len({r['source_sha256'] for r in finalized}) == len(finalized)
    assert len({r[s] for r in finalized for s in ['styled', 'destyled']}) == 124
    out = HERE / 'role_signal_pairs.jsonl'
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in finalized).encode()
    out.write_bytes(payload)
    source_hashes = set()
    assignment_path = HERE / 'outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/split-assignment.json'
    assignment = json.loads(assignment_path.read_text())
    for key in assignment['domains']['c4_dolma3']:
        if key.startswith('article:'): source_hashes.add(key.split(':', 1)[1])
    hashes = {hashlib.sha256(r[s].encode()).hexdigest() for r in finalized for s in ['styled', 'destyled']}
    audit = dict(format_version='role-signal-controlled-pairs-v1', freeze_date='2026-11-12',
                 pairs_sha256=hashlib.sha256(payload).hexdigest(), counts=dict(split_counts),
                 family_counts=dict(family_counts), semantic_units=62, text_variants=124,
                 minimum_tokens=min(min(r['token_lengths'].values()) for r in finalized),
                 maximum_tokens=max(max(r['token_lengths'].values()) for r in finalized),
                 tokenizer=str(TOKENIZER), all_content_preserved=True, all_conclusions_preserved=True,
                 probe_scores_seen=False, exact_full_text_hash_matches_with_article_source_cluster_hashes=len(hashes & source_hashes),
                 article_source_cluster_hashes_checked=len(source_hashes),
                 overlap_limitation='Hash comparison checks exact complete-text equality against available C4/Dolma3 source-cluster identities. It cannot exclude shared phrases, corpus near-duplicates, or model-pretraining exposure. No raw source text was copied; all semantic records were newly authored.',
                 dependencies='20 formal reasoning families; each has 2 or 3 formal semantic units with distinct facts. The ten pilot units share families, but never source facts or text, with formal units. Two construction units are separate sources. All texts share controlled style scaffolds; family-cluster sensitivity is recommended.',
                 legacy_pairs=dict(base_cache='repro/cache/chat_01_base_policies.json', base_count=626,
                                   destyled_cache='repro/cache/chat_01_destyled_policies.json', destyled_count=300,
                                   generator='google/gemini-2.5-pro',
                                   excluded_reason='Existing StrongREJECT synthetic attack-policy pairs were found, but benign controlled texts were selected for the main experiment. Historical destyled text also retains policy and we/should cues.'))
    (HERE / 'role_signal_pairs-audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
