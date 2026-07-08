// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Select from 'antd/lib/select';
import Space from 'antd/lib/space';
import Text from 'antd/lib/typography/Text';
import { CloseOutlined } from '@ant-design/icons';

import { CombinedState } from 'reducers';
import { changeSelectedAnnotationsLabelAsync } from 'actions/annotation-actions';

function MultiSelectPanel(): JSX.Element | null {
    const dispatch = useDispatch();

    const selectedStateIDs = useSelector(
        (state: CombinedState) => state.annotation.annotations.selectedStateIDs,
    );
    const labels = useSelector(
        (state: CombinedState) => state.annotation.job.labels,
    );
    const canvasInstance = useSelector(
        (state: CombinedState) => state.annotation.canvas.instance,
    );

    if (!selectedStateIDs || selectedStateIDs.length === 0) {
        return null;
    }

    const handleLabelChange = (labelID: number): void => {
        const label = labels.find((l: any) => l.id === labelID);
        if (label) {
            dispatch(changeSelectedAnnotationsLabelAsync(label) as any);
        }
    };

    const handleClear = (): void => {
        if (canvasInstance && typeof (canvasInstance as any).clearMultiselection === 'function') {
            (canvasInstance as any).clearMultiselection();
        }
    };

    return (
        <Alert
            className='cvat-multiselect-panel'
            type='info'
            message={(
                <Space direction='vertical' size={4} style={{ width: '100%' }}>
                    <Space>
                        <Text strong>{`${selectedStateIDs.length} object${selectedStateIDs.length > 1 ? 's' : ''} selected`}</Text>
                        <Button
                            size='small'
                            icon={<CloseOutlined />}
                            onClick={handleClear}
                            title='Clear selection'
                        />
                    </Space>
                    <Space>
                        <Text>Change label:</Text>
                        <Select
                            size='small'
                            placeholder='Select label'
                            style={{ minWidth: 120 }}
                            onChange={handleLabelChange}
                        >
                            {labels.map((label: any) => (
                                <Select.Option key={label.id} value={label.id}>
                                    {label.name}
                                </Select.Option>
                            ))}
                        </Select>
                    </Space>
                </Space>
            )}
        />
    );
}

export default React.memo(MultiSelectPanel);
