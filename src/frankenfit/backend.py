# Unpublished Copyright (c) 2022 Max Bane, all rights reserved.
#
# NOTICE: All information contained herein is, and remains the property of Max Bane.
# The intellectual and technical concepts contained herein are proprietary to Max Bane
# and may be covered by U.S. and Foreign Patents, patents in process, and are protected
# by trade secret or copyright law. Dissemination of this information or reproduction
# of this material is strictly forbidden unless prior written permission is obtained
# from Max Bane. Access to the source code contained herein is hereby forbidden to
# anyone except current employees, contractors, or customers of Max Bane who have
# executed Confidentiality and Non-disclosure agreements explicitly covering such
# access.
#
# The copyright notice above does not evidence any actual or intended publication or
# disclosure of this source code, which includes information that is confidential
# and/or proprietary, and is a trade secret, of Max Bane. ANY REPRODUCTION,
# MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE
# OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN CONSENT OF MAX BANE IS STRICTLY
# PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE
# RECEIPT OR POSSESSION OF THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY
# OR IMPLY ANY RIGHTS TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, NOR TO
# MANUFACTURE, USE, OR SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from frankenfit.core import FitTransform, Transform


class Backend(ABC):
    @abstractmethod
    def fit(
        self,
        transform: Transform,
        data_fit: Optional[object] = None,
        bindings: Optional[dict[str, object]] = None,
    ) -> FitTransform:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, fit_transform: FitTransform, data_apply: Optional[object] = None
    ) -> object:
        raise NotImplementedError


class LocalBackend(Backend):
    def fit(
        self,
        transform: Transform,
        data_fit: Optional[object] = None,
        bindings: Optional[dict[str, object]] = None,
    ) -> FitTransform:
        return transform.fit(data_fit=data_fit, bindings=bindings)

    def apply(
        self, fit_transform: FitTransform, data_apply: Optional[object] = None
    ) -> object:
        return fit_transform.apply(data_apply)
