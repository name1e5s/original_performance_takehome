"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import random
from typing import DefaultDict
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

ENGINES = ["alu", "valu", "load", "store", "flow", "debug"]
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
    "debug": 64,
}

@dataclass
class Instruction:
    id: int
    engine: str
    slot: tuple

    depends_on: int
    war_depends_by: list[int]
    depends_by: list[int]

    depends_on_list: list[int]

    def op(self):
        return self.slot[0]
    
    def addrs_for_alu(self, *slot):
        assert self.engine == "alu"
        (op, dest, a1, a2) = slot
        return [dest], [a1, a2]
    
    def addrs_for_valu(self, *slot):
        addrs_to_write = []
        addrs_to_read = []

        assert self.engine == "valu"
        match slot:
            case ("vbroadcast", dest, src):
                for i in range(VLEN):
                    addrs_to_write.append(dest + i)
                addrs_to_read.append(src)
            case ("multiply_add", dest, a, b, c):
                for i in range(VLEN):
                    addrs_to_write.append(dest + i)
                    addrs_to_read.append(a + i)
                    addrs_to_read.append(b + i)
                    addrs_to_read.append(c + i)
            case (op, dest, a1, a2):
                for i in range(VLEN):
                    addrs_to_write.append(dest + i)
                    addrs_to_read.append(a1 + i)
                    addrs_to_read.append(a2 + i)
            case _:
                raise NotImplementedError(f"Unknown valu op {slot}")
        
        return addrs_to_write, addrs_to_read

    def addrs_for_load(self, *slot):
        addrs_to_write = []
        addrs_to_read = []

        assert self.engine == "load"
        match slot:
            case ("load", dest, addr):
                # print(dest, addr, core.scratch[addr])
                addrs_to_write.append(dest)
                addrs_to_read.append(addr)
            case ("load_offset", dest, addr, offset):
                addrs_to_write.append(dest + offset)
                addrs_to_read.append(addr + offset)
            case ("vload", dest, addr):  # addr is a scalar
                addrs_to_read.append(addr)
                for vi in range(VLEN):
                    addrs_to_write.append(dest + vi)
            case ("const", dest, val):
                addrs_to_write.append(dest)
            case _:
                raise NotImplementedError(f"Unknown load op {slot}")

        return addrs_to_write, addrs_to_read

    def addrs_for_store(self, *slot):
        addrs_to_write = []
        addrs_to_read = []
        match slot:
            case ("store", addr, src):
                addrs_to_read.append(addr)
                addrs_to_read.append(src)
            case ("vstore", addr, src):  # addr is a scalar
                addrs_to_read.append(addr)
                for vi in range(VLEN):
                    addrs_to_read.append(src + vi)
            case _:
                raise NotImplementedError(f"Unknown store op {slot}")
        
        return addrs_to_write, addrs_to_read

    def addrs_for_flow(self, *slot):
        addrs_to_write = []
        addrs_to_read = []

        match slot:
            case ("select", dest, cond, a, b):
                addrs_to_write.append(dest)
                addrs_to_read.append(cond)
                addrs_to_read.append(a)
                addrs_to_read.append(b)
            case ("add_imm", dest, a, imm):
                addrs_to_write.append(dest)
                addrs_to_read.append(a)
            case ("vselect", dest, cond, a, b):
                for vi in range(VLEN):
                    addrs_to_write.append(dest + vi)
                    addrs_to_read.append(cond + vi)
                    addrs_to_read.append(a + vi)
                    addrs_to_read.append(b + vi)
            case ("halt",):
                # no-op
                pass
            case ("pause",):
                # no-op
                pass
            case ("trace_write", val):
                addrs_to_read.append(val)
            case ("cond_jump", cond, addr):
                addrs_to_read.append(cond)
            case ("cond_jump_rel", cond, offset):
                addrs_to_read.append(cond)
            case ("jump", addr):
                pass
            case ("jump_indirect", addr):
                addrs_to_read.append(addr)
            case ("coreid", dest):
                addrs_to_write.append(dest)
            case _:
                raise NotImplementedError(f"Unknown flow op {slot}")

        return addrs_to_write, addrs_to_read
    
    def addrs_for_debug(self, *slot):
        addrs_to_write = []
        addrs_to_read = []
        match slot:
            case ("compare", loc, key):
                addrs_to_read.append(loc)
            case ("vcompare", loc, keys):
                for vi in range(VLEN):
                    addrs_to_read.append(loc + vi)
            case ("comment", _):
                pass
            case _:
                pass
        return addrs_to_write, addrs_to_read

    def addrs(self):
        match self.engine:
            case "alu":
                return self.addrs_for_alu(*self.slot)
            case "valu":
                return self.addrs_for_valu(*self.slot)
            case "load":
                return self.addrs_for_load(*self.slot)
            case "store":
                return self.addrs_for_store(*self.slot)
            case "flow":
                return self.addrs_for_flow(*self.slot)
            case "debug":
                return self.addrs_for_debug(*self.slot)
            case _:
                raise NotImplementedError(f"Unknown engine {self.engine}")


@dataclass
class ScoreBoard:
    last_reads: set = field(default_factory=set)
    last_write: int = -1

class InstructionScheduler:
    def __init__(self):
        self.scoreboard = defaultdict(ScoreBoard)
        self.instructions: list[Instruction] = []
    
    def append(self, instr_raw: tuple[Engine, tuple]):
        instr = Instruction(
            id=len(self.instructions),
            engine=instr_raw[0],
            slot=instr_raw[1],
            depends_on=0,
            war_depends_by=[],
            depends_by=[],
            depends_on_list=[],
        )
        index = len(self.instructions)
        self.instructions.append(instr)

        addrs_to_write, addrs_to_read = instr.addrs()

        war_deps: set[int] = set()
        other_deps: set[int] = set()

        for addr in addrs_to_write:
            last_reads = self.scoreboard[addr].last_reads
            # write after read dependency
            for last_read in last_reads:
                war_deps.add(last_read)

            last_write = self.scoreboard[addr].last_write
            # write after write dependency
            if last_write > -1:
                other_deps.add(last_write)

        for addr in addrs_to_read:
            last_write = self.scoreboard[addr].last_write
            # read after write dependency
            if last_write > -1:
                other_deps.add(last_write)

        for dep in war_deps:
            instr.depends_on += 1
            instr.depends_on_list.append(dep)
            self.instructions[dep].war_depends_by.append(index)

        for dep in other_deps:
            instr.depends_on += 1
            instr.depends_on_list.append(dep)
            self.instructions[dep].depends_by.append(index)
        
        for addr in addrs_to_read:
            self.scoreboard[addr].last_reads.add(index)

        for addr in addrs_to_write:
            self.scoreboard[addr].last_write = index
            self.scoreboard[addr].last_reads.clear()

        return index
    
    def extend(self, instrs: list[tuple[Engine, tuple]]):
        for instr in instrs:
            self.append(instr)
    
    def schedule(self) -> list[dict[str, list(tuple)]]:
        if not self.instructions:
            return []

        current_time = 0
        instructions_to_schedule = len(self.instructions)
        bundles: list[dict[str, list(tuple)]] = []
        # engine -> heap[(ready_at, instr_id)]
        ready: dict[str, list[tuple[int, int]]] = defaultdict(list)
        scheduled = set()

        for instr in self.instructions:
            if instr.depends_on == 0:
                heapq.heappush(ready[instr.engine], (current_time, instr.id))
        
        while instructions_to_schedule > 0:
            bundle: dict[str, list(tuple)] = {}
            slot_count: dict[str, int] = defaultdict(int)
            can_pack_more = True

            for engine in ENGINES:
                bundle[engine] = []

            while can_pack_more:
                scheduled_this_cycle: list[int] = []

                for engine in ENGINES:
                    while (slot_count[engine] < SLOT_LIMITS[engine] and
                        ready[engine]):
                        ready_at, instr_id = heapq.heappop(ready[engine])
                        if ready_at > current_time:
                            heapq.heappush(ready[engine], (ready_at, instr_id))
                            break

                        instr = self.instructions[instr_id]
                        bundle[engine].append(instr.slot)
                        if instr_id in scheduled:
                            print(f"Warning: Instruction {instr_id} already scheduled")
                            exit(1)
                        scheduled.add(instr_id)
                        scheduled_this_cycle.append(instr_id)
                        slot_count[engine] += 1
                        instructions_to_schedule -= 1
                
                # move to next cycle
                if not scheduled_this_cycle:
                    can_pack_more = False
                    break
                
                war_deps: list[int] = []
                other_deps: list[int] = []
                for instr_id in scheduled_this_cycle:
                    instr = self.instructions[instr_id]
                    war_deps.extend(instr.war_depends_by)
                    other_deps.extend(instr.depends_by)

                deps = [(dep_id, 0) for dep_id in war_deps]
                deps.extend([(dep_id, 1) for dep_id in other_deps])
                for (dep_id, delay) in deps:
                    dep_instr = self.instructions[dep_id]
                    dep_instr.depends_on -= 1
                    if dep_instr.depends_on == 0:
                        heapq.heappush(ready[dep_instr.engine], (current_time + delay, dep_id))

            bundles.append(bundle)
            current_time += 1

        print(f"Scheduled {len(self.instructions)} instructions in {current_time} cycles {len(scheduled)}")

        return bundles


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage_start"))))
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        const_vlen = self.scratch_const(VLEN)
        const_double_vlen = self.scratch_const(VLEN * 2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # body = []  # array of slots
        body = InstructionScheduler()

        # Scalar scratch registers
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        v_batch_size = batch_size // VLEN
        # store indices and values in scratch
        indices = self.alloc_scratch("indices", batch_size)
        values = self.alloc_scratch("values", batch_size)

        tmp_addr_0 = self.alloc_scratch("tmp_addr_0")
        tmp_addr_1 = self.alloc_scratch("tmp_addr_1")


        body.append(("flow", ("add_imm", tmp_addr_0, self.scratch["inp_values_p"], 0)))
        body.append(("flow", ("add_imm", tmp_addr_1, self.scratch["inp_values_p"], VLEN)))
        for i in range(0, v_batch_size, 2):
            offset_0 = i * VLEN
            offset_1 = offset_0 + VLEN
            body.append(("load", ("vload", values + offset_0, tmp_addr_0)))
            body.append(("load", ("vload", values + offset_1, tmp_addr_1)))
            body.append(("alu", ("+", tmp_addr_0, tmp_addr_0, const_double_vlen)))
            body.append(("alu", ("+", tmp_addr_1, tmp_addr_1, const_double_vlen)))

        for round in range(rounds):
            for i in range(batch_size):
                tmp_idx = indices + i
                tmp_val = values + i
                # idx = mem[inp_indices_p + i]
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.append(("debug", ("compare", tmp_val, (round, i, "tmp_val"))))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))

        body.append(("flow", ("add_imm", tmp_addr_0, self.scratch["inp_indices_p"], 0)))
        body.append(("flow", ("add_imm", tmp_addr_1, self.scratch["inp_values_p"], 0)))
        for i in range(v_batch_size):
            offset = i * VLEN
            body.append(("store", ("vstore", tmp_addr_0, indices + offset)))
            body.append(("store", ("vstore", tmp_addr_1, values + offset)))
            body.append(("alu", ("+", tmp_addr_0, tmp_addr_0, const_vlen)))
            body.append(("alu", ("+", tmp_addr_1, tmp_addr_1, const_vlen)))

        #body_instrs = self.build(body.schedule())
        self.instrs.extend(body.schedule())
        # body_instrs = self.build(body)
        # self.instrs.extend(body_instrs)

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
