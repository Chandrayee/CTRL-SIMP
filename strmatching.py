import difflib
def show_diff(seqm):
    """Unify operations between two compared strings seqm is a difflib.SequenceMatcher instance whose a & b are strings"""
    output= []
    for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
        if opcode == 'equal':
            output.append(" ".join(seqm.a[a0:a1]))
        elif opcode == 'insert':
            output.append(" <ins>" + " ".join(seqm.b[b0:b1]) + "</ins> ")
        elif opcode == 'delete':
            output.append(" <del>" + " ".join(seqm.a[a0:a1]) + "</del> ")
        elif opcode == 'replace':
            output.append(" <rep>" + " ".join(seqm.a[a0:a1]) + "<by>" + " ".join(seqm.b[b0:b1]) +  "</rep> ")
        else:
            output.append("....")
    return ''.join(output)

'''Use:    diff_str = difflib.SequenceMatcher(None, original_sentence, rewritten_sentence)
        diff_anno = show_diff(diff_str)
'''

def grouped_matcher(l1, l2):
    seq_mat = difflib.SequenceMatcher(a=l1, b=l2, autojunk=True)
    l3 = list(l1)
    for groups in seq_mat.get_grouped_opcodes(n=40):
        for operation, i1,i2,j1,j2 in groups:
            if operation == "delete":
                print("Deleting Sequence : '{}' from l1".format(l1[i1:i2]))
                l3[i1:i2] = [""] * len(l1[i1:i2])
            elif operation == "replace":
                print("Replacing Sequence : '{}' in l1 with '{}' in l2".format(l1[i1:i2], l2[j1:j2]))
                l3[i1:i2] = l2[j1:j2]
            elif operation == "insert":
                print("Inserting Sequence : '{}' from l2 at {} in l1".format(l2[j1:j2], i1))
                l3.insert(i1, l2[j1:j2])
            elif operation == "equal":
                print("Equal Sequences. '{}' No Action Needed.".format(l1[i1:i2]))

    print("\nFinal Sequence : {}".format("".join(l3)))

def get_codes(a, b):
    s = difflib.SequenceMatcher(None, a, b, autojunk=True)
    for tag, i0, i1, j0, j1 in s.get_opcodes():
        print('{:7} a[{}:{}] ---> b[{}:{}] {!r:>8} --> {!r}'.format(
            tag, i0, i1, j0, j1, " ".join(a[i0:i1]), " ".join(b[j0:j1])
        ))