"""Optional length-based read decision for 1Cat-vLLM v1.5.3's Mooncake Store."""

import logging

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
)

logger = logging.getLogger("vllm.task_a_routing")


class LengthRoutedMooncakeStoreConnector(MooncakeStoreConnector):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        matched, asynchronous = super().get_num_new_matched_tokens(request, num_computed_tokens)
        policy = (request.kv_transfer_params or {}).get("read_prefix_lengths")
        if matched and policy is not None:
            read = policy == "all" or matched in policy
            logger.info("Task A routing request=%s matched=%d decision=%s",
                        request.request_id, matched, "read" if read else "compute")
            if not read:
                # The parent prepared this load but has not allocated its slots
                # or issued the GET. Keep normal recording and save behavior.
                self.connector_scheduler.load_specs.pop(request.request_id)
                return 0, False
        return matched, asynchronous
