// LLVM pass plugin: instruments every load/store and basic block so the
// memtrace runtime (memtrace_rt.c) can model private caches and emit an
// L2-miss stream in Ramulator SimpleO3 trace format.
//
//   clang -O2 -fpass-plugin=libMemTracePass.so ... memtrace_rt.o
//
// Inserted calls:
//   __mt_bb(i32 n)              at each basic-block entry (n = instructions in block)
//   __mt_ld(ptr p, i32 bytes)   before each load
//   __mt_st(ptr p, i32 bytes)   before each store
// Atomic RMW / cmpxchg are recorded as stores. Memory intrinsics (memcpy,
// memset, memmove) are recorded as a read and/or write of the full range.
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"

using namespace llvm;

namespace {

struct MemTracePass : PassInfoMixin<MemTracePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    LLVMContext &C = M.getContext();
    const DataLayout &DL = M.getDataLayout();
    Type *I32 = Type::getInt32Ty(C);
    Type *I64 = Type::getInt64Ty(C);
    Type *Ptr = PointerType::getUnqual(C);
    Type *Void = Type::getVoidTy(C);
    FunctionCallee BB = M.getOrInsertFunction("__mt_bb", Void, I32);
    FunctionCallee LD = M.getOrInsertFunction("__mt_ld", Void, Ptr, I32);
    FunctionCallee ST = M.getOrInsertFunction("__mt_st", Void, Ptr, I32);
    FunctionCallee RNG = M.getOrInsertFunction("__mt_range", Void, Ptr, I64, I32);

    for (Function &F : M) {
      if (F.isDeclaration() || F.getName().starts_with("__mt_")) continue;
      for (BasicBlock &B : F) {
        SmallVector<Instruction *, 32> Targets;
        unsigned Count = 0;
        for (Instruction &I : B) {
          if (isa<DbgInfoIntrinsic>(I) || isa<PHINode>(I)) continue;
          ++Count;
          if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<AtomicRMWInst>(I) ||
              isa<AtomicCmpXchgInst>(I) || isa<MemIntrinsic>(I))
            Targets.push_back(&I);
        }
        if (Count == 0) continue;
        IRBuilder<> Entry(&*B.getFirstInsertionPt());
        Entry.CreateCall(BB, {ConstantInt::get(I32, Count)});

        for (Instruction *I : Targets) {
          IRBuilder<> IRB(I);
          auto Size = [&](Type *T) {
            return ConstantInt::get(I32, (uint32_t)DL.getTypeStoreSize(T).getKnownMinValue());
          };
          if (auto *L = dyn_cast<LoadInst>(I)) {
            IRB.CreateCall(LD, {L->getPointerOperand(), Size(L->getType())});
          } else if (auto *S = dyn_cast<StoreInst>(I)) {
            IRB.CreateCall(ST, {S->getPointerOperand(), Size(S->getValueOperand()->getType())});
          } else if (auto *A = dyn_cast<AtomicRMWInst>(I)) {
            IRB.CreateCall(ST, {A->getPointerOperand(), Size(A->getValOperand()->getType())});
          } else if (auto *X = dyn_cast<AtomicCmpXchgInst>(I)) {
            IRB.CreateCall(ST, {X->getPointerOperand(), Size(X->getNewValOperand()->getType())});
          } else if (auto *MI = dyn_cast<MemIntrinsic>(I)) {
            Value *Len = IRB.CreateZExtOrTrunc(MI->getLength(), I64);
            if (auto *MT = dyn_cast<MemTransferInst>(MI))
              IRB.CreateCall(RNG, {MT->getSource(), Len, ConstantInt::get(I32, 0)});
            IRB.CreateCall(RNG, {MI->getDest(), Len, ConstantInt::get(I32, 1)});
          }
        }
      }
    }
    return PreservedAnalyses::none();
  }
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MemTrace", "0.1", [](PassBuilder &PB) {
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase) {
                  MPM.addPass(MemTracePass());
                });
          }};
}
