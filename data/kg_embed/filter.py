ents = set()
with open("entity2id.txt") as fin:
    fin.readline()
    for line in fin:
        qid, _ = line.split("\t")
        ents.add(qid)
with open("web_entity.txt") as fin:
    with open("entity_map.txt", "w") as fout:
        for line in fin:
            _, qid = line.strip().split("\t")
            if qid in ents:
                fout.write(line)
