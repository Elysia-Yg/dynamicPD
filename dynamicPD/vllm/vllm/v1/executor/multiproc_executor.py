from typing import Any

from vllm.logger import logger
from vllm.v1.outputs import (ModelRunnerOutput,LogprobsLists, KVConnectorOutput)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorStats)
from vllm_ascend.worker.model_runner_v1 import AsyncNPUModelRunnerOutput


from vllm.v1.executor.multiproc_executor import  MultiprocExecutor, WorkerProc

from dynamicPD.patching import dynamicPDPatch

class MultiprocExecutorPatch(dynamicPDPatch[MultiprocExecutor]):
    def profile_npu(self, is_start: bool = True) -> None:
        self.collective_rpc("profile_npu", args=(is_start, ), unique_reply_rank=self.output_rank)

class WorkerProcPatch(dynamicPDPatch[WorkerProc]):
    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncNPUModelRunnerOutput): #AsyncNPUModelRunnerOutput
            decode_output = None
            prefill_output = None
            logger.debug(f"decode_output is None: {output._model_runner_output is None}, prefill_output is None: {output._prefill_model_runner_output is None}")
            if output._model_runner_output is not None:
                decode_output = output.get_output()
            if output._prefill_model_runner_output is not None:
                prefill_output = output._prefill_model_runner_output  # output.get_prefill_output()   
            logger.debug("get output over")
            if isinstance(decode_output, Exception) or isinstance(prefill_output, Exception):
                if isinstance(decode_output, Exception):
                    result = (WorkerProc.ResponseStatus.FAILURE, str(decode_output))
                else:
                    result = (WorkerProc.ResponseStatus.FAILURE, str(prefill_output))
            elif decode_output is None or prefill_output is None or decode_output.req_ids == [] or prefill_output.req_ids == []:
                if decode_output is None or decode_output.req_ids == []:
                    merged_output = prefill_output
                    merged_output.is_merged = True
                    merged_output.finished_prefill_reqs = prefill_output.req_ids
                    result = (WorkerProc.ResponseStatus.SUCCESS, merged_output)
                else:
                    merged_output = decode_output
                    result = (WorkerProc.ResponseStatus.SUCCESS, merged_output)
            else:
                logger.debug(f"decode_output req_ids: {decode_output.req_ids}, prefill_output req_ids: {prefill_output.req_ids}")
                merged_output = self.merge_enqueue_output(decode_output, prefill_output)
                result = (WorkerProc.ResponseStatus.SUCCESS, merged_output)
            if (response_mq := self.worker_response_mq) is not None:
                response_mq.enqueue(result)
        else:
            if isinstance(output, Exception):
                result = (WorkerProc.ResponseStatus.FAILURE, str(output))
            else:
                result = (WorkerProc.ResponseStatus.SUCCESS, output)
            if (response_mq := self.worker_response_mq) is not None:
                response_mq.enqueue(result)
            
    def merge_enqueue_output(self,decode_output: ModelRunnerOutput,
                             prefill_output: ModelRunnerOutput) -> ModelRunnerOutput:
        if decode_output is None or prefill_output is None:
            if decode_output is not None:
                output = decode_output
            else:
                output = prefill_output
        else:
            output = ModelRunnerOutput(
                req_ids=None,
                req_id_to_index=None,
                sampled_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict=None,
                pooler_output=None,
            )
            
            output.finished_prefill_reqs = prefill_output.req_ids
            output.req_ids = decode_output.req_ids + prefill_output.req_ids
            output.sampled_token_ids = decode_output.sampled_token_ids + prefill_output.sampled_token_ids
            output.pooler_output = decode_output.pooler_output + prefill_output.pooler_output
            
            output.req_id_to_index = {req_id: idx for idx, req_id in enumerate(output.req_ids)}
            
            output.prompt_logprobs_dict = {**decode_output.prompt_logprobs_dict, **prefill_output.prompt_logprobs_dict}
            
            if decode_output.num_nans_in_logits is not None or prefill_output.num_nans_in_logits is not None:
                output.num_nans_in_logits = {}
                if decode_output.num_nans_in_logits is not None:
                    output.num_nans_in_logits.update(decode_output.num_nans_in_logits)
                if prefill_output.num_nans_in_logits is not None:
                    output.num_nans_in_logits.update(prefill_output.num_nans_in_logits)
                    
            if decode_output.logprobs is not None or prefill_output.logprobs is not None:
                lp1 = decode_output.logprobs if decode_output.logprobs is not None else LogprobsLists([], [], [])
                lp2 = prefill_output.logprobs if prefill_output.logprobs is not None else LogprobsLists([], [], [])
                output.logprobs = LogprobsLists(
                    logprob_tokne_ids = lp1.logprob_tokne_ids + lp2.logprob_tokne_ids,
                    logprobs= lp1.logprobs + lp2.logprobs,
                    sampled_token_ranks= lp1.sampled_token_ranks + lp2.sampled_token_ranks
                )
            else:
                output.logprobs = None
                
            if decode_output.kv_connector_output is not None or prefill_output.kv_connector_output is not None:
                kv1 = decode_output.kv_connector_output
                kv2 = prefill_output.kv_connector_output   
                merged_kv = KVConnectorOutput()
                
                def merge_sets(s1, s2):
                    if s1 is None and s2 is None:
                        return None
                    result = set()
                    if s1 is not None:
                        result.update(s1)
                    if s2 is not None:
                        result.update(s2)
                    return result

                merged_kv.finished_sending = merge_sets(getattr(kv1, 'finished_sending', None),getattr(kv2, 'finished_sending', None))
                merged_kv.finished_recving = merge_sets(getattr(kv1, 'finished_recving', None),getattr(kv2, 'finished_recving', None))
                stats1 = getattr(kv1, 'kv_connector_stats', None)
                stats2 = getattr(kv2, 'kv_connector_stats', None)
                if stats1 is not None and stats2 is not None:
                    merged_stats = KVConnectorStats()
                    merged_stats.data = {**(getattr(stats1, 'data', {})), **(getattr(stats2, 'data', {}))}
                    merged_kv.kv_connector_stats = merged_stats
                    
                output.kv_connector_output = merged_kv
            else:
                output.kv_connector_output = None
                
            logger.debug(f"is_merged: True, req_ids: {output.req_ids}")
            output.is_merged = True
        return output
        
    def destroy_model_parallel(self):
        from vllm.distributed.parallel_state import _offload_TP, _DP, _EP, _PP, _TP, _DCP
        if _offload_TP:
            _offload_TP.destroy()
        _offload_TP = None
        if _DP:
            _DP.destroy()
        _DP = None
        if _EP:
            _EP.destroy()
        _EP = None
        if _PP:
            _PP.destroy()
        _PP = None
        if _TP:
            _TP.destroy()
        _TP = None
        if _DCP:
            _DCP.destroy()
        _DCP = None

    def shutdown(self):
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        self.destroy_model_parallel()
        destroy_distributed_environment()
        